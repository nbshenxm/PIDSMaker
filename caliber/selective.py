"""Risk-coverage analysis of abstention criteria (selective prediction for PIDS alerts).

The budget controller's alert set is held fixed; each criterion ranks the alerts by how
uncertain the automatic decision is, and the top fraction f is abstained on (sent to an
analyst instead of auto-alerted). We report the precision of the remaining automatic alerts
as a function of f, per criterion. This isolates the quality of the uncertainty RANKING from
budget dynamics. Escalation is the mirror image: rank sub-threshold units by how likely their
decision is to flip and send the top k to analysts.

Criteria (higher = more uncertain):
  ens_ci       -(S - q) / sem      ensemble margin in standard errors (training randomness)
  raw_margin   -(S - q)            score margin only (no uncertainty)
  single_run   -(z_0 - q)          margin of one ensemble member (a single-run PIDS)
  vote_split   1 - |2v - 1|        v = fraction of members above q
  spread       sem                 raw member disagreement
  unfamiliar   label unseen among benign validation entities (ties by raw margin)
  random

  python -m caliber.selective --artifact_dir A --system velox --dataset CADETS_E3
"""

import argparse
import json
import os

import numpy as np

from caliber import data
from caliber.budget import replay_budget
from caliber.online import ensemble_stream

FRACS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def criteria(S, Z, q, unfamiliar, rng):
    M = Z.shape[0]
    sem = Z.std(0) / np.sqrt(M) + 1e-9
    v = (Z >= q[None, :]).mean(0)
    return {
        "ens_ci": -(S - q) / sem,
        "raw_margin": -(S - q),
        "single_run": -(Z[0] - q),
        "vote_split": 1 - np.abs(2 * v - 1),
        "spread": sem,
        "unfamiliar": unfamiliar.astype(float) * 1e6 - (S - q),
        "random": rng.random(len(S)),
    }


def unfamiliar_mask(members, meta, nodes, cache):
    if meta is None:
        return np.zeros(len(nodes), bool)
    norm = meta["path"].astype(str).str.replace(r"\d+", "#", regex=True)
    path = dict(zip(meta["node"].astype(np.int64), norm))
    m = members[0]
    ep = data.select_epoch(m, "min_val_loss", cache)
    v = data.read_split(m["edge_losses_dir"], "val", ep, cache)
    seen = {path.get(n) for n in set(v["srcnode"].astype(np.int64)) | set(v["dstnode"].astype(np.int64))}
    return np.array([path.get(int(n)) not in seen for n in nodes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", default="da39a3ee5e")
    ap.add_argument("--budget", type=float, default=50)
    args = ap.parse_args()
    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    members = data.group_manifests(data.load_manifests(
        args.artifact_dir, args.system, args.dataset, args.group))[(args.system, args.dataset, args.group)]
    labels, meta = data.labels_for(members[0], args.artifact_dir)
    st = ensemble_stream(members, labels, cache_dir=cache)
    B = args.budget
    cal = st["S_val"][np.isfinite(st["S_val"])]
    nvd = max(len(members[0].get("val_dates", [])), 1)
    k = min(max(int(B * nvd), 1), len(cal))
    q0, eta = float(np.sort(cal)[-k]), 0.02 * (float(np.std(cal)) or 1.0)
    f, inc, q = replay_budget(st["S"], st["t1"], st["w"], st["y"], st["node"], B, q0, eta,
                              return_q=True)
    y, node, S, Z = st["y"], st["node"], st["S"], st["Z"]
    unf = unfamiliar_mask(members, meta, node, cache)
    rng = np.random.default_rng(0)
    out = {"alerts": int(f.sum()), "attack_alerts": int((f & y).sum())}
    print(f"== {args.system}/{args.dataset} B={B} alerts={f.sum()} attack alerts={(f & y).sum()} "
          f"base precision={(f & y).sum() / max(f.sum(), 1):.3f}")

    # --- abstention on alerts: precision of the remaining automatic alerts vs fraction abstained
    a_idx = np.flatnonzero(f)
    crit = criteria(S[a_idx], Z[:, a_idx], q[a_idx], unf[a_idx], rng)
    out["abstain"] = {}
    for name, c in crit.items():
        order = np.argsort(-c, kind="stable")              # most uncertain first
        row = {}
        for fr in FRACS:
            n_abs = int(round(fr * len(order)))
            keep = order[n_abs:]
            abst = order[:n_abs]
            prec = float(y[a_idx[keep]].mean()) if len(keep) else float("nan")
            prec_abs = float(y[a_idx[abst]].mean()) if len(abst) else float("nan")
            row[str(fr)] = {"auto_precision": prec, "abstained_precision": prec_abs,
                            "attack_entities_auto": len(set(node[a_idx[keep]][y[a_idx[keep]]]))}
        row["mean_auto_precision"] = float(np.nanmean([row[str(fr)]["auto_precision"] for fr in FRACS]))
        out["abstain"][name] = row
        print(f"  abstain {name:11s} " + " ".join(
            f"{fr:.2f}:{row[str(fr)]['auto_precision']:.3f}" for fr in FRACS)
            + f" | mean={row['mean_auto_precision']:.3f}")

    # --- escalation below threshold: attack units among the top-k escalated
    b_idx = np.flatnonzero(~f)
    sem_b = Z[:, b_idx].std(0) / np.sqrt(Z.shape[0]) + 1e-9
    gap = q[b_idx] - S[b_idx]
    esc = {"ens_ci": gap / sem_b, "raw_margin": gap, "single_run": q[b_idx] - Z[0, b_idx],
           "random": rng.random(len(b_idx))}
    out["escalate"] = {}
    for name, c in esc.items():
        order = np.argsort(c, kind="stable")                # closest to flipping first
        row = {}
        for kk in [int(B), int(2 * B), int(5 * B), int(10 * B)]:
            top = b_idx[order[:kk]]
            row[str(kk)] = {"attack_units": int(y[top].sum()),
                            "attack_entities": len(set(node[top][y[top]]))}
        out["escalate"][name] = row
        print(f"  escalate {name:10s} " + " ".join(
            f"top{kk}:{row[str(kk)]['attack_entities']}" for kk in [int(B), int(2 * B), int(5 * B), int(10 * B)]))
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    with open(os.path.join(rep, f"selective_{args.system}_{args.dataset}_{args.group}_B{int(B)}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
