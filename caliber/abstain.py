"""When should a PIDS abstain? Trigger study at fixed analyst capacity.

A PIDS should abstain (route a decision to an analyst instead of auto-alerting or
auto-dismissing) when its own decision would be unreliable. We test one trigger per measured
source of unreliability, alone and combined, against the plain budget controller:

  ci        training randomness: |S - q| < k * sd_members / sqrt(M)   (another run could flip it)
  tie       score plateaus: unit lies in a tie block (>= T units within eps of its score in the
            same hour) straddling the threshold -> one grouped case per block
  unfamiliar out-of-support entity: its label/path never occurs among benign validation entities
  lineage   controller-level: incidents are counted per (lineage, hour) and analysts see distinct
            lineages first (flood defense; see lineage_study)
  combined  all of the above

Every abstained case is an incident in the same analyst queue as alerts, so the budget
controller (and Proposition 1) covers alerts + abstentions. Analysts review at most B cases
per day (FIFO, lineage-first when lineage is on). Metrics:
  attack entities reviewed (seen by an analyst), benign cases/day vs B, automatic-alert
  precision, and automatic-decision agreement between two disjoint half-ensembles.

  python -m caliber.abstain --artifact_dir A --system velox --dataset CADETS_E3
"""

import argparse
import json
import os

import numpy as np

from caliber import data
from caliber.budget import DAY, HOUR
from caliber.lineage import lineage_keys, load_parents, process_predicate
from caliber.online import day_of, ensemble_stream


def tie_blocks(S, hour, eps):
    """Size of the tie block each unit belongs to: units in the same hour within eps."""
    key = np.round(S / eps).astype(np.int64)
    pair = hour.astype(np.int64) * 10**7 + (key - key.min())
    _, inv, cnt = np.unique(pair, return_inverse=True, return_counts=True)
    return cnt[inv], pair


def replay_abstain(st, idx, keys, B, q0, eta, triggers, unfamiliar, k_ci=2.0, tie_T=20,
                   tie_eps=None, max_gap_ns=HOUR, rho=None):
    """Returns decision codes per unit (0 dismiss, 1 alert, 2 abstain) and case flags
    (unit opened a new case = incident).

    rho: abstention share of the budget. At most rho*B new abstention cases per day, served
    most-uncertain first (smallest |S - q| / sem); candidates beyond the share fall back to the
    plain threshold (alert if S >= q, else dismiss). None = unlimited."""
    Z = st["Z"][idx]
    S = Z.mean(0)
    sem = Z.std(0) / np.sqrt(Z.shape[0])
    t, w, y = st["t1"], st["w"], st["y"]
    hour = t // HOUR
    tie_size, tie_id = tie_blocks(S, hour, tie_eps) if "tie" in triggers else (None, None)
    order = np.lexsort((t, w))
    dec = np.zeros(len(S), np.int8)
    case = np.zeros(len(S), bool)
    last = {}
    q, clock = q0, None
    day_id, day_abst = None, 0
    bounds = np.flatnonzero(np.diff(w[order])) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(order)]):
        ix = order[lo:hi]
        tmax = t[ix].max()
        dt = 0.0 if clock is None else min(max(tmax - clock, 0), max_gap_ns) / DAY
        clock = tmax if clock is None else max(clock, tmax)
        if int(clock // DAY) != day_id:
            day_id, day_abst = int(clock // DAY), 0
        s = S[ix]
        above = s >= q
        abst = np.zeros(len(ix), bool)
        if "ci" in triggers:
            abst |= np.abs(s - q) < k_ci * np.maximum(sem[ix], 1e-9)
        if "tie" in triggers:
            # a big tie block straddling the threshold: its members cannot be ranked
            near = np.abs(s - q) < tie_eps
            abst |= near & (tie_size[ix] >= tie_T)
        if "unfamiliar" in triggers:
            abst |= above & unfamiliar[ix]
        if rho is not None and abst.any():
            # serve the abstention share most-uncertain first; overflow -> plain threshold
            cand = np.flatnonzero(abst)
            margin = np.abs(s[cand] - q) / np.maximum(sem[ix][cand], 1e-9)
            cand = cand[np.argsort(margin, kind="stable")]
            keep = np.zeros(len(ix), bool)
            for j in cand:
                kj = keys[ix[j]]
                open_case = kj in last and t[ix[j]] - last[kj] < HOUR
                if open_case or day_abst < rho * B:
                    keep[j] = True
                    if not open_case:
                        day_abst += 1
            abst = keep
        alert = above & ~abst
        dec[ix[alert]] = 1
        dec[ix[abst & (above | ("ci" in triggers) | ("tie" in triggers))]] = 2
        n_new = 0
        n_att = 0
        for j in np.flatnonzero(dec[ix] > 0):
            i = ix[j]
            # tie blocks open one case per block; others one case per (key, hour)
            kk = ("tie", tie_id[i]) if (dec[i] == 2 and "tie" in triggers
                                        and tie_size[i] >= tie_T) else keys[i]
            if kk in last and t[i] - last[kk] < HOUR:
                continue
            last[kk] = t[i]
            case[i] = True
            n_new += 1
            n_att += int(y[i])
        q += eta * (n_new - n_att - B * dt)            # attacks refunded (verdict delay: see budget.py)
    return dec, case


def review(st, dec, case, keys, B, tz, lineage_first, priority=None):
    """Analysts review at most B cases/day: FIFO, cases flagged in `priority` first (e.g.
    unfamiliar entities), then distinct lineages first if lineage_first."""
    days = day_of(st["t1"], tz)
    t, y, node = st["t1"], st["y"], st["node"]
    hour = t // HOUR
    seen = set()
    for d in np.unique(days):
        ix = np.flatnonzero(case & (days == d))
        ix = ix[np.argsort(t[ix], kind="stable")]
        if priority is not None:
            ix = np.r_[ix[priority[ix]], ix[~priority[ix]]]
        if lineage_first:
            first, rest, s_ = [], [], set()
            for i in ix:
                (rest if keys[i] in s_ else first).append(i)
                s_.add(keys[i])
            ix = np.array(first + rest, dtype=np.int64)
        for i in ix[: int(B)]:
            seen.add((keys[i], hour[i]))
            if dec[i] == 2:
                seen.add(("u", i))
    acted = (dec > 0) & y
    att_seen = {node[i] for i in np.flatnonzero(acted)
                if (keys[i], hour[i]) in seen or ("u", i) in seen}
    nd = max(len(np.unique(days)), 1)
    auto = dec == 1
    return {"attack_entities_reviewed": len(att_seen), "attack_entities": len(set(node[y])),
            "attack_entities_auto": len(set(node[auto & y])),
            "attack_entities_abstained": len(set(node[(dec == 2) & y])),
            "benign_cases_per_day": float((case & ~y).sum() / nd),
            "abstain_cases_per_day": float((case & (dec == 2)).sum() / nd),
            "auto_precision": float((auto & y).sum() / max(auto.sum(), 1))}


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


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
    tz = labels.get("timezone", "US/Eastern")
    nvd = max(len(members[0].get("val_dates", [])), 1)
    M = st["Z"].shape[0]
    B = args.budget
    parents = load_parents(args.artifact_dir, args.dataset)
    if parents is not None:
        lin = lineage_keys(st["node"], st["partner"], parents, process_predicate(parents, meta), 1)
    else:
        lin = st["node"].astype(np.int64)
    ent = st["node"].astype(np.int64)
    # unfamiliar: entity label never seen among validation entities. Labels are normalized
    # (digits -> '#') so per-PID paths such as /proc/1234/stat match /proc/#/stat.
    unfamiliar = np.zeros(len(ent), bool)
    if meta is not None:
        norm = meta["path"].astype(str).str.replace(r"\d+", "#", regex=True)
        path = dict(zip(meta["node"].astype(np.int64), norm))
        val_nodes = set()
        for m in members[:1]:
            ep = data.select_epoch(m, "min_val_loss", cache)
            v = data.read_split(m["edge_losses_dir"], "val", ep, cache)
            val_nodes = set(v["srcnode"].astype(np.int64)) | set(v["dstnode"].astype(np.int64))
        seen_paths = {path.get(n) for n in val_nodes}
        unfamiliar = np.array([path.get(n) not in seen_paths for n in ent.tolist()])
    cal_full = np.nanmean(st["Z_val"], 0)
    tie_eps = 0.01 * (float(np.nanstd(cal_full)) or 1.0)
    halves = {"A": np.arange(0, M // 2), "B": np.arange(M // 2, 2 * (M // 2)), "all": np.arange(M)}
    # (triggers, abstention share rho, review priority)
    policies = {
        "none": ([], None, None),
        "ci rho=.25": (["ci"], 0.25, None), "ci rho=.5": (["ci"], 0.5, None),
        "ci unlimited": (["ci"], None, None),
        "tie rho=.25": (["tie"], 0.25, None),
        "unfamiliar-abstain rho=.25": (["unfamiliar"], 0.25, None),
        "unfamiliar-first review": ([], None, "unfamiliar"),
        "lineage": (["lineage"], None, None),
        "ci+lineage rho=.25": (["ci", "lineage"], 0.25, None),
        "ci+lineage rho=.5": (["ci", "lineage"], 0.5, None),
        "ci+lineage rho=.25 +unfamiliar-first": (["ci", "lineage"], 0.25, "unfamiliar"),
    }
    print(f"== {args.system}/{args.dataset} M={M} B={B} unfamiliar units={unfamiliar.mean():.3%} "
          f"lineage={'yes' if parents is not None else 'no'}")
    out = {}
    for name, (trig, rho, prio) in policies.items():
        keys = lin if "lineage" in trig else ent
        res = {}
        autos = {}
        for h, idx in halves.items():
            cal = np.nanmean(st["Z_val"][idx], 0)
            cal = cal[np.isfinite(cal)]
            k = min(max(int(B * nvd), 1), len(cal))
            q0, eta = float(np.sort(cal)[-k]), 0.02 * (float(np.std(cal)) or 1.0)
            dec, case = replay_abstain(st, idx, keys, B, q0, eta, trig, unfamiliar,
                                       tie_eps=tie_eps, rho=rho)
            if h == "all":
                res = review(st, dec, case, keys, B, tz, "lineage" in trig,
                             priority=unfamiliar if prio == "unfamiliar" else None)
            # agreement on automatic CASES: (key, hour), key = entity or lineage
            m = (dec == 1)
            autos[h] = set(zip(keys[m].tolist(), (st["t1"][m] // HOUR).tolist()))
        res["auto_agreement_halves"] = jaccard(autos["A"], autos["B"])
        out[name] = res
        print(f"  {name:38s} reviewed={res['attack_entities_reviewed']:3d}/{res['attack_entities']} "
              f"(auto {res['attack_entities_auto']}, abstained {res['attack_entities_abstained']}) "
              f"benign cases/day={res['benign_cases_per_day']:6.1f} abstain/day={res['abstain_cases_per_day']:5.1f} "
              f"auto-precision={res['auto_precision']:.3f} auto-agreement(A,B)={res['auto_agreement_halves']:.3f}",
              flush=True)
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    with open(os.path.join(rep, f"abstain_{args.system}_{args.dataset}_{args.group}_B{int(B)}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
