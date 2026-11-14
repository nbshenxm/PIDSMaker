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
from caliber.lineage import case_keys
from caliber.online import day_of, ensemble_stream
from caliber.tiers import (BUCKET_H, KAPPA, _with_borrower, case_threshold, replay_tiers, review_step,
                           unit_attacks)

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
    case_att = [set() for _ in range(n)]
    for i, ks in unit_attacks(node, t, idx[y[idx]], attacks).items():
        case_att[case_of[i]].update(ks)
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
    case_att = [set() for _ in range(n)]
    for i, ks in unit_attacks(st["node"], st["t1"], np.flatnonzero(st["y"]), attacks).items():
        case_att[inv[i]].update(ks)
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
    elif policy == "hourtopB":                      # online top-k: each hour's best B/24 cases,
        hours = (open_t // HOUR).astype(np.int64)   # released at the end of that hour
        keep = np.zeros(len(score), bool)
        release = open_t.copy()
        credit = 0.0
        for h in np.unique(hours):
            ix = np.flatnonzero(hours == h)
            credit += B / 24.0                       # fractional quota carries over; idle
            k = int(credit)                          # capacity does not accumulate
            credit -= k
            keep[ix[np.argsort(-score[ix])[:k]]] = True
            release[ix] = (h + 1) * HOUR
        return release[keep], score[keep], [case_att[i] for i in np.flatnonzero(keep)]
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


def allowlist_scores(nodes, times, artifact_dir, dataset, meta):
    """Allowlist detector (after Guerra et al.): a process is suspicious if its executable name or
    path was not seen during training. As an allowlist is operated, an executable alerts when it
    first appears and is allowlisted once reviewed, so only the first unit of each unseen path
    carries the novelty score; later units keep only the rarity term. Graded so that a budget can
    be applied: unseen name (2) + unseen path (1) + rarity of the name in training
    (1/(1+count)). Non-process units never alert (-inf); the original scores processes only."""
    import gzip
    import pandas as pd
    from collections import Counter
    p = os.path.join(artifact_dir, "caliber", "labels", f"{dataset}__train_procs.csv.gz")
    tp = pd.read_csv(p, dtype={"path": str}).fillna("")
    base = lambda s: s.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    name_cnt = Counter(base(x) for x in tp["path"])
    train_paths = set(tp["path"])
    sub = meta[meta["type"] == "subject"]
    path_of = dict(zip(sub["node"].astype(np.int64), sub["path"].fillna("").astype(str)))
    out = np.full(len(nodes), -np.inf)
    seen = set()                                    # paths already alerted on (allowlisted)
    nodes = nodes.astype(np.int64)
    for i in np.argsort(times, kind="stable"):
        n = nodes[i]
        if n not in path_of:
            continue
        path = path_of[n]
        nm = base(path)
        novelty = 2.0 * (nm not in name_cnt) + 1.0 * (path not in train_paths)
        if novelty and path in seen:
            novelty = 0.0
        elif novelty:
            seen.add(path)
        out[i] = novelty + 1.0 / (1 + name_cnt[nm])
    return out


def one_deployment(S, S_val, st, keys, B, nvd, policy, attacks, tz, cand=None,
                   verdict_delay_ns=8 * HOUR, max_gap_ns=HOUR, keys_val=None, bucket_h=BUCKET_H,
                   U=None, U_val=None, views=None):
    """policy: static | prevday | daytopB | hourtopB | hindsight | caliber (token bucket) |
    caliber_cap (earlier per-batch flood cap) | caliber_nocap (controller only).
    Every policy starts from the validation threshold that would have opened B cases per day
    (keys_val: case keys of the validation units). CALIBER policies review on the ensemble upper
    bound U (= S for a single run); the baselines use the detector score S, as deployed today."""
    if policy.startswith("caliber") and U is not None:
        S, S_val = U, U_val
    cal = S_val[np.isfinite(S_val)]
    q0 = case_threshold(S_val, st["t_val"], st["node_val"] if keys_val is None else keys_val, B, nvd)
    n_flood = 0
    summary = None                                  # per case: is it a summary (bulk) item
    if policy in ("hindsight", "prevday", "daytopB", "hourtopB"):
        open_t, score, case_att = fixed_threshold_cases(cand, policy, q0, B)
    else:
        eta = 0.0 if policy == "static" else review_step(cal, B)
        if policy == "caliber":
            kw = {"bucket_h": bucket_h}
        else:                                       # static: as deployed today, no cap
            kw = {"flood_cap": "auto" if policy == "caliber_cap" else None}
        if views is not None and policy.startswith("caliber"):
            # online round-robin: one controller per view, B/K each, analysts get the union
            from caliber.multiview import replay_views
            K = len(views)
            kv = st["node_val"] if keys_val is None else keys_val
            q0s = [case_threshold(Uv, st["t_val"], kv, B / K, nvd) for _, Uv in views]
            etas = [review_step(Uv[np.isfinite(Uv)], B / K) for _, Uv in views]
            case_of, cases = replay_views([Ut for Ut, _ in views], st["t1"], st["w"], st["y"],
                                          keys, B, q0s, etas, np.random.default_rng(0),
                                          refund_delay_ns=verdict_delay_ns, max_gap_ns=max_gap_ns,
                                          **kw)
        else:
            case_of, cases = replay_tiers(S, st["t1"], st["w"], st["y"], keys, B, q0, eta, 2.0, 0.0,
                                          0.0, np.inf, np.random.default_rng(0),
                                          refund_delay_ns=verdict_delay_ns, max_gap_ns=max_gap_ns,
                                          **kw)
        n_flood = sum(bool(c.get("flood")) for c in cases)
        open_t, score, case_att = cases_of(case_of, cases, S, st["t1"], st["node"], st["y"], attacks)
        summary = np.array([bool(c.get("flood")) for c in cases], bool)
        for c, cs in enumerate(cases):              # token bucket: summary delivered later/never
            if "t_deliver" in cs:
                open_t[c] = np.inf if cs["t_deliver"] is None else float(cs["t_deliver"])
    days = day_of(open_t[np.isfinite(open_t)], tz) if len(open_t) else np.array([])
    # days with data (a few spill-over units across midnight do not make a monitored day)
    all_days, n_units = np.unique(day_of(st["t1"], tz), return_counts=True)
    test_days = all_days[n_units >= 0.05 * np.median(n_units)]
    per_day = np.array([np.sum(days == d) for d in test_days], float)
    reviewed = analyst_queue(open_t, score, B, float(st["t1"].max()))
    present = [k_ for k_, a in enumerate(attacks) if set(a["nids"]) & set(st["node"][st["y"]])]
    ttr, ttr_ind = {}, {}
    for c in np.flatnonzero(np.isfinite(reviewed)):
        for a in case_att[c]:
            h = (reviewed[c] - attacks[a]["start_ns"]) / HOUR
            ttr[a] = min(ttr.get(a, np.inf), h)
            if summary is None or not summary[c]:   # seen in an individual case, not a summary
                ttr_ind[a] = min(ttr_ind.get(a, np.inf), h)
    # attack cases never reviewed (expired in the queue): the replay refunded them at the verdict
    # delay, which overstates the refunds an analyst could have given; should stay ~0
    unrev = sum(1 for c in range(len(case_att)) if case_att[c] and not np.isfinite(reviewed[c]))
    return {"load_ratio": float(per_day.mean() / B), "worst_day_ratio": float(per_day.max() / B),
            "flood_cases": int(n_flood), "unreviewed_attack_cases": int(unrev),
            "attacks": len(present), "attacks_seen": sum(a in ttr for a in present),
            "attacks_seen_individual": sum(a in ttr_ind for a in present),
            "time_to_review_h": {str(a): float(ttr[a]) for a in present if a in ttr},
            "time_to_review_individual_h": {str(a): float(ttr_ind[a]) for a in present
                                            if a in ttr_ind}}


def inject_decoys(st, Z, keys, attacks, mode, rate, hours, seed=0):
    """Decoy flood (squealing, Patton et al. RAID'01): from each attack's start, the attacker
    raises `rate` decoy alerts per hour for `hours` hours, each scoring above every unit of that
    attack (worst case: decoys outrank the attack in every run). mode 'one': all decoys come from
    the attack's own lineage (one foothold); 'many': each decoy from a distinct lineage."""
    rng = np.random.default_rng(seed)
    yi = np.flatnonzero(st["y"])
    ua = unit_attacks(st["node"], st["t1"], yi, attacks)
    order = np.argsort(st["t1"], kind="stable")
    ts, ws = st["t1"][order], st["w"][order]
    add_t, add_w, add_k, add_z = [], [], [], []
    for k, a in enumerate(attacks):
        units = np.array([i for i, ks in ua.items() if k in ks], np.int64)
        if len(units) == 0:
            continue
        zmax = np.nanmax(Z[:, units], 0)
        z = float(np.nanmax(zmax)) + 0.01
        n = int(round(rate * hours))
        tt = (a["start_ns"] + rng.uniform(0, hours * HOUR, n)).astype(np.int64)
        pos = np.clip(np.searchsorted(ts, tt, side="right") - 1, 0, len(ts) - 1)
        add_t.append(tt)
        add_w.append(ws[pos])
        if mode == "one":
            add_k.append(np.full(n, keys[units[np.argmax(zmax)]], np.int64))
        else:
            add_k.append(-(np.arange(n, dtype=np.int64) + 1 + 10**12 + k * 10**7))
        add_z.append(np.full(n, z))
    if not add_t:
        return st, Z, keys
    t_new, w_new = np.concatenate(add_t), np.concatenate(add_w)
    n_new = len(t_new)
    st2 = dict(st)
    st2["t1"] = np.concatenate([st["t1"], t_new])
    st2["w"] = np.concatenate([st["w"], w_new])
    st2["y"] = np.concatenate([st["y"], np.zeros(n_new, bool)])
    st2["node"] = np.concatenate([st["node"], -(np.arange(n_new, dtype=np.int64) + 1 + 2 * 10**12)])
    Z2 = np.hstack([Z, np.tile(np.concatenate(add_z)[None, :], (Z.shape[0], 1))])
    return st2, Z2, np.concatenate([keys.astype(np.int64), np.concatenate(add_k)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", default="da39a3ee5e")
    ap.add_argument("--subsets", type=int, default=5, help="random ensembles per size")
    ap.add_argument("--policies",
                    default="static,prevday,daytopB,hourtopB,hindsight,caliber,caliber_cap,caliber_nocap")
    ap.add_argument("--bucket_h", type=float, default=BUCKET_H, help="token bucket depth (hours of B)")
    ap.add_argument("--decoys", default="", help="decoy flood 'mode:rate:hours', mode one|many")
    ap.add_argument("--gap_cap_h", type=float, default=1.0, help="credited time per data gap")
    ap.add_argument("--budgets", default="", help="comma list (default: all)")
    ap.add_argument("--tag", default="", help="suffix for the report file")
    ap.add_argument("--detector", default="model", choices=["model", "allowlist"],
                    help="allowlist: executable-novelty baseline on the same units (Guerra et al.)")
    ap.add_argument("--scorer", default="ensemble", choices=["ensemble", "abstain", "plain3"],
                    help="ensemble: robust-z mean of the group's runs (default); abstain: the "
                         "abstention-aware scorer (caliber.abstain_score) on density runs; plain3: "
                         "its plain-surprise channel alone on the same units and seeds")
    ap.add_argument("--seeds", default="0,1,2", help="seeds for --scorer abstain/plain3")
    args = ap.parse_args()
    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    members = data.group_manifests(data.load_manifests(
        args.artifact_dir, args.system, args.dataset, args.group))[(args.system, args.dataset, args.group)]
    labels, meta = data.labels_for(members[0], args.artifact_dir)
    labels = _with_borrower(labels, members[0], args.artifact_dir)
    if args.scorer == "ensemble":
        st = ensemble_stream(members, labels, cache_dir=cache)
    else:
        from caliber import abstain_score
        st = abstain_score.scored_stream(args.scorer, args.artifact_dir, args.system, args.dataset,
                                         labels, seeds=tuple(int(s) for s in args.seeds.split(",")))
    tz = labels.get("timezone", "US/Eastern")
    keys, has_lin = case_keys(st, args.artifact_dir, args.dataset, meta)
    keys_val, _ = case_keys(st, args.artifact_dir, args.dataset, meta, split="val")
    print(f"cases by {'lineage' if has_lin else 'entity'}", flush=True)
    nvd = max(len(members[0].get("val_dates", [])), 1)
    Z, Zv, M = st["Z"], st["Z_val"], st["Z"].shape[0]
    if args.detector == "allowlist":
        # same units as the learned detector, scored by executable novelty (one "member")
        Z = allowlist_scores(st["node"], st["t1"], args.artifact_dir, args.dataset, meta)[None, :]
        Zv = allowlist_scores(st["node_val"], st["t_val"], args.artifact_dir, args.dataset, meta)[None, :]
        M = 1
    if args.decoys:
        mode, rate, hours = args.decoys.split(":")
        st, Z, keys = inject_decoys(st, Z, keys, labels["attacks"], mode, float(rate), float(hours))
        print(f"decoys: {args.decoys} -> {len(st['t1'])} units", flush=True)
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
    for B in ([int(b) for b in args.budgets.split(",")] if args.budgets else BUDGETS):
        for kind, subset in configs:
            S = np.nanmean(Z[subset], 0)
            S_val = np.nanmean(Zv[subset], 0)
            if args.scorer != "ensemble" and len(subset) == M and M > 1:
                # the pooled scorer is not the mean of the per-run scores; its interval is
                # computed in abstain_score (U = S + KAPPA * sd over runs / sqrt(R))
                S, S_val, U, U_val = st["S"], st["S_val"], st["U"], st["U_val"]
            elif len(subset) > 1:                   # review statistic: upper bound over runs
                U = S + KAPPA * np.nan_to_num(np.nanstd(Z[subset], 0) / np.sqrt(len(subset)))
                U_val = S_val + KAPPA * np.nan_to_num(np.nanstd(Zv[subset], 0) / np.sqrt(len(subset)))
            else:
                U, U_val = S, S_val
            views = None
            key = tuple(subset)
            if key not in cand_cache:
                cand_cache[key] = candidate_cases(S, st, keys, labels["attacks"], tz)
            for policy in args.policies.split(","):
                r = one_deployment(S, S_val, st, keys, B, nvd, policy, labels["attacks"], tz,
                                   cand_cache[key], max_gap_ns=int(args.gap_cap_h * HOUR),
                                   keys_val=keys_val, bucket_h=args.bucket_h, U=U, U_val=U_val, views=views)
                rows.append({"B": B, "runs": kind, "members": subset, "policy": policy, **r})
        print(f"B={B} done", flush=True)
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    tag = args.system if args.detector == "model" else "allowlist"
    sc = "" if args.scorer == "ensemble" else f"_{args.scorer}"
    out = os.path.join(rep, f"deploy_{tag}_{args.dataset}_{args.group}{sc}{args.tag}.json")
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
