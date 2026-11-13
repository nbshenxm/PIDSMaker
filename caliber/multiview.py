"""Online round-robin of several score views: one budget controller per view.

The abstention-aware scorer ranks cases by a round-robin of two views (S = b + q + mlvmax and
C = b + q). Its online form runs one review controller per view with budget B/K each, on that
view's own score scale (no cross-view calibration, which drifts between validation and test),
and the analysts receive the union of the views' cases. Each controller satisfies Proposition 1
for its B/K, and a case alerted by several views is delivered once, so the union receives at
most B benign cases per day up to the same additive constants. The act tier (with its own
false-automation budget) runs on one view; the others only review.
"""
import numpy as np

from caliber.tiers import replay_tiers


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def merge_cases(results):
    """results: [(case_of, cases)] per view over the same units. Cases of different views that
    share a unit become one case (union-find). Returns (case_of, cases) like replay_tiers."""
    offs = np.cumsum([0] + [len(c) for _, c in results])
    parent = list(range(offs[-1]))
    n_units = len(results[0][0])
    first = np.full(n_units, -1, np.int64)
    for k, (case_of, _) in enumerate(results):
        ix = np.flatnonzero(case_of >= 0)
        gid = case_of[ix] + offs[k]
        has = first[ix] >= 0
        for a, b in zip(first[ix][has], gid[has]):
            ra, rb = _find(parent, int(a)), _find(parent, int(b))
            if ra != rb:
                parent[rb] = ra
        first[ix[~has]] = gid[~has]
    roots = np.array([_find(parent, i) for i in range(offs[-1])], np.int64)
    uniq, new_id = np.unique(roots, return_inverse=True)
    merged = [None] * len(uniq)
    for k, (_, cases) in enumerate(results):
        for j, c in enumerate(cases):
            m = new_id[offs[k] + j]
            cur = merged[m]
            if cur is None:
                merged[m] = dict(c)
                continue
            cur["tier"] = "A" if ("A" in (cur["tier"], c["tier"])) else "B"
            cur["audited"] = cur["audited"] or c["audited"]
            cur["y"] = cur["y"] or c["y"]
            cur["flood"] = bool(cur.get("flood")) and bool(c.get("flood"))
            # delivery: an individual case is delivered when it opens; summary cases later/never
            if "t_deliver" not in c or "t_deliver" not in cur:
                cur.pop("t_deliver", None)
            else:
                ts = [x for x in (cur["t_deliver"], c["t_deliver"]) if x is not None]
                cur["t_deliver"] = min(ts) if ts else None
    for c in merged:                                # a bulk/summary item is never automated
        if c.get("flood"):
            c["tier"] = "B"
    case_of = np.where(first >= 0, new_id[np.maximum(first, 0)], -1)
    return case_of, merged


def replay_views(Ss, t, w, y, keys, B, q0s, etas, rng, act_view=0, act=None, **common):
    """Ss: unit scores per view; q0s, etas: warm start and step per view, both computed for the
    view's budget B/len(Ss). act: replay_tiers act-tier arguments (pi, audit, eta_act, q_act0,
    B_auto, S_act, act_key, act_seen) applied to view act_view. common: max_gap_ns, flood_cap,
    refund_delay_ns, bucket_h."""
    K = len(Ss)
    res = []
    for k, S in enumerate(Ss):
        a = dict(act or {}) if k == act_view else {}
        pi, audit = a.pop("pi", 2.0), a.pop("audit", 0.0)
        eta_act, q_act0 = a.pop("eta_act", 0.0), a.pop("q_act0", np.inf)
        res.append(replay_tiers(S, t, w, y, keys, B / K, q0s[k], etas[k], pi, audit, eta_act,
                                q_act0, rng, **a, **common))
    return merge_cases(res)
