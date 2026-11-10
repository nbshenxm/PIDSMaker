"""Deployment study: many replayed deployments of one detector, with an analyst queue.

A deployment = (analyst budget B, the training runs the operator happens to have, policy).
From the scores already computed we replay every combination of
  B in BUDGETS, a single run (each member) or an ensemble of m members (random subsets),
  policy 'static' (threshold fixed on validation, as PIDS are deployed today) or
  'caliber' (label-free budget controller, Proposition 1),
and feed the resulting cases (lineage x hour) to an analyst queue: capacity B cases per day,
spread over the day, highest score first, unreviewed cases carried over for up to
MAX_WAIT_DAYS.

Per deployment we record
  load_ratio      mean daily cases / B, and the worst day's ratio
  attacks_seen    attacks whose first case was reviewed (attack level, as in Orthrus/ADP)
  time_to_review  hours from attack start to the first review of one of its cases

  python -m caliber.deploy_study --artifact_dir A --system velox --dataset THEIA_E3
"""

import argparse
import heapq
import json
import os

import numpy as np

from caliber import data
from caliber.budget import DAY, HOUR
from caliber.lineage import lineage_keys, load_parents, process_predicate
from caliber.online import day_of, ensemble_stream
from caliber.tiers import _with_borrower, replay_tiers

BUDGETS = [20, 50, 100, 200, 500]
MAX_WAIT_DAYS = 3


def cases_of(case_of, cases, S, t, node, y, attacks):
    """Per case: open time, priority score, the attacks it contains."""
    n = len(cases)
    open_t = np.full(n, np.inf)
    score = np.full(n, -np.inf)
    idx = np.flatnonzero(case_of >= 0)
    np.minimum.at(open_t, case_of[idx], t[idx])
    np.maximum.at(score, case_of[idx], S[idx])
    att_nodes = [set(a["nids"]) for a in attacks]
    case_att = [set() for _ in range(n)]
    for i in idx[y[idx]]:
        for k, ns in enumerate(att_nodes):
            if node[i] in ns:
                case_att[case_of[i]].add(k)
    return open_t, score, case_att


def analyst_queue(open_t, score, B, t_end):
    """Review times (ns, inf if never) under capacity B/day, highest score first."""
    order = np.argsort(open_t)
    reviewed = np.full(len(open_t), np.inf)
    if len(order) == 0:
        return reviewed
    per_hour, credit, heap, j = B / 24.0, 0.0, [], 0
    h = int(open_t[order[0]] // HOUR)
    h_end = int(t_end // HOUR) + 24 * MAX_WAIT_DAYS
    while h <= h_end and (j < len(order) or heap):
        while j < len(order) and open_t[order[j]] < (h + 1) * HOUR:
            c = order[j]
            heapq.heappush(heap, (-score[c], open_t[c], c))
            j += 1
        credit += per_hour
        while heap and credit >= 1.0:
            _, ot, c = heapq.heappop(heap)
            if (h + 1) * HOUR - ot > MAX_WAIT_DAYS * DAY:
                continue                    # expired unreviewed
            reviewed[c] = (h + 1) * HOUR
            credit -= 1.0
        if not heap:
            credit = min(credit, per_hour)  # idle capacity does not accumulate
        h += 1
    return reviewed


def candidate_cases(S, st, keys, attacks, tz):
    """All (lineage, hour) cases with their best score: under any threshold that is fixed for a
    day, a case alerts exactly when its best unit reaches the threshold."""
    hour = (st["t1"] // HOUR).astype(np.int64)
    _, inv = np.unique(np.stack([keys.astype(np.int64), hour]), axis=1, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    score = np.full(n, -np.inf)
    open_t = np.full(n, np.inf)
    np.maximum.at(score, inv, S)
    np.minimum.at(open_t, inv, st["t1"])
    att_nodes = [set(a["nids"]) for a in attacks]
    case_att = [set() for _ in range(n)]
    for i in np.flatnonzero(st["y"]):
        for k, ns in enumerate(att_nodes):
            if st["node"][i] in ns:
                case_att[inv[i]].add(k)
    return score, open_t, case_att, day_of(open_t, tz)


def fixed_threshold_cases(cand, policy, q0, B):
    """Cases alerted by a threshold that changes at most once per day."""
    score, open_t, case_att, days = cand
    if policy == "static":
        q_of_day = {d: q0 for d in np.unique(days)}
    elif policy == "hindsight":                     # label-free, but uses the whole test period
        nd = len(np.unique(days))
        k = min(max(int(B * nd), 1), len(score))
        q_of_day = {d: float(np.sort(score)[-k]) for d in np.unique(days)}
    elif policy == "prevday":                       # yesterday's B-th best case score
        q_of_day, prev = {}, q0
        for d in sorted(np.unique(days)):
            q_of_day[d] = prev
            s = np.sort(score[days == d])
            prev = float(s[-min(int(B), len(s))]) if len(s) else prev
    elif policy == "daytopB":                       # Ho et al. '17: the day's top B, at day end
        keep = np.zeros(len(score), bool)
        release = open_t.copy()
        for d in np.unique(days):
            ix = np.flatnonzero(days == d)
            top = ix[np.argsort(-score[ix])[:int(B)]]
            keep[top] = True
            release[ix] = open_t[ix].max()           # ranked once the day's data are in
        return release[keep], score[keep], [case_att[i] for i in np.flatnonzero(keep)]
    else:
        raise ValueError(policy)
    keep = score >= np.array([q_of_day[d] for d in days])
    return open_t[keep], score[keep], [case_att[i] for i in np.flatnonzero(keep)]


def one_deployment(S, S_val, st, keys, B, nvd, policy, attacks, tz, cand=None):
    cal = S_val[np.isfinite(S_val)]
    k = min(max(int(B * nvd), 1), len(cal))
    q0 = float(np.sort(cal)[-k])
    if policy in ("hindsight", "prevday", "daytopB"):
        open_t, score, case_att = fixed_threshold_cases(cand, policy, q0, B)
    else:
        eta = 0.0 if policy == "static" else 0.02 * (float(np.std(cal)) or 1.0)
        case_of, cases = replay_tiers(S, st["t1"], st["w"], st["y"], keys, B, q0, eta, 2.0, 0.0,
                                      0.0, np.inf, np.random.default_rng(0))
        open_t, score, case_att = cases_of(case_of, cases, S, st["t1"], st["node"], st["y"], attacks)
    days = day_of(open_t[np.isfinite(open_t)], tz) if len(open_t) else np.array([])
    test_days = np.unique(day_of(st["t1"], tz))
    per_day = np.array([np.sum(days == d) for d in test_days], float)
    reviewed = analyst_queue(open_t, score, B, float(st["t1"].max()))
    present = [k_ for k_, a in enumerate(attacks) if set(a["nids"]) & set(st["node"][st["y"]])]
    ttr = {}
    for c in np.flatnonzero(np.isfinite(reviewed)):
        for a in case_att[c]:
            ttr[a] = min(ttr.get(a, np.inf), (reviewed[c] - attacks[a]["start_ns"]) / HOUR)
    return {"load_ratio": float(per_day.mean() / B), "worst_day_ratio": float(per_day.max() / B),
            "attacks": len(present), "attacks_seen": sum(a in ttr for a in present),
            "time_to_review_h": {str(a): float(ttr[a]) for a in present if a in ttr}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", default="da39a3ee5e")
    ap.add_argument("--subsets", type=int, default=5, help="random ensembles per size")
    ap.add_argument("--policies", default="static,prevday,daytopB,hindsight,caliber")
    args = ap.parse_args()
    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    members = data.group_manifests(data.load_manifests(
        args.artifact_dir, args.system, args.dataset, args.group))[(args.system, args.dataset, args.group)]
    labels, meta = data.labels_for(members[0], args.artifact_dir)
    labels = _with_borrower(labels, members[0], args.artifact_dir)
    st = ensemble_stream(members, labels, cache_dir=cache)
    tz = labels.get("timezone", "US/Eastern")
    parents = load_parents(args.artifact_dir, args.dataset)
    keys = (lineage_keys(st["node"], st["partner"], parents, process_predicate(parents, meta), 1)
            if parents is not None else st["node"].astype(np.int64))
    nvd = max(len(members[0].get("val_dates", [])), 1)
    Z, Zv, M = st["Z"], st["Z_val"], st["Z"].shape[0]
    rng = np.random.default_rng(0)
    configs = [("single", [m]) for m in range(M)]
    for size in sorted({3, 5, M}):
        if size > M:
            continue
        draws = 1 if size == M else args.subsets
        configs += [(f"ensemble{size}", sorted(rng.choice(M, size, replace=False).tolist()))
                    for _ in range(draws)]
    rows = []
    cand_cache = {}
    for B in BUDGETS:
        for kind, subset in configs:
            S = np.nanmean(Z[subset], 0)
            S_val = np.nanmean(Zv[subset], 0)
            key = tuple(subset)
            if key not in cand_cache:
                cand_cache[key] = candidate_cases(S, st, keys, labels["attacks"], tz)
            for policy in args.policies.split(","):
                r = one_deployment(S, S_val, st, keys, B, nvd, policy, labels["attacks"], tz,
                                   cand_cache[key])
                rows.append({"B": B, "runs": kind, "members": subset, "policy": policy, **r})
        print(f"B={B} done", flush=True)
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    out = os.path.join(rep, f"deploy_{args.system}_{args.dataset}_{args.group}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    summarize(rows, f"{args.system}/{args.dataset}")


def summarize(rows, title):
    """Compact view: per (runs kind, policy) over budgets and subsets."""
    print(f"== {title}")
    kinds = sorted({(r["runs"], r["policy"]) for r in rows})
    for kind, policy in kinds:
        rs = [r for r in rows if r["runs"] == kind and r["policy"] == policy]
        lr = np.array([r["load_ratio"] for r in rs])
        wd = np.array([r["worst_day_ratio"] for r in rs])
        seen = np.array([r["attacks_seen"] / max(r["attacks"], 1) for r in rs])
        ttr = [v for r in rs for v in r["time_to_review_h"].values()]
        print(f"  {kind:10s} {policy:8s} n={len(rs):3d} load/B median {np.median(lr):.2f} "
              f"[p10 {np.percentile(lr, 10):.2f}, p90 {np.percentile(lr, 90):.2f}] "
              f"P(worst day>2B)={np.mean(wd > 2):.2f} | attacks seen mean {seen.mean():.2f} "
              f"p10 {np.percentile(seen, 10):.2f} | time to review median "
              f"{np.median(ttr) if ttr else float('nan'):.1f} h", flush=True)


if __name__ == "__main__":
    main()
