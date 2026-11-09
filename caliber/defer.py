"""Split-vote deferral and decision reproducibility (goal G1).

Each unit gets an ensemble score S and member votes: member m "votes alert" when its
normalized score z_m >= q (members are standardized, so one threshold applies to all).
  alert   S >= q and every member votes alert (unanimous)
  defer   some but not all members vote alert, within the deferral share rho*B per day
          (served in decreasing vote order); beyond the share: alert if S >= q else dismiss
  dismiss otherwise
Alerts and deferrals are both incidents for the budget controller.

Reproducibility: split the M members into two disjoint halves A and B, run the policy on
each half independently (own ensemble, own controller), and compare their AUTOMATIC
incident sets (entity, hour) with the Jaccard index. Baselines: single runs (pairs of
members) and the ensemble without deferral.

  python -m caliber.defer --artifact_dir A --system velox --dataset CADETS_E3
"""

import argparse
import json
import os

import numpy as np

from caliber import data
from caliber.budget import DAY, HOUR
from caliber.online import day_of, ensemble_stream


def replay_defer(S, Z, t, w, y, node, B, q0, eta, rho=0.0, dedupe_ns=HOUR, max_gap_ns=HOUR):
    """Returns per-unit decision codes: 0 dismiss, 1 alert, 2 defer; and incident flags."""
    order = np.lexsort((t, w))
    S, Z, t, w, y, node = S[order], Z[:, order], t[order], w[order], y[order], node[order]
    dec = np.zeros(len(S), np.int8)
    inc = np.zeros(len(S), bool)
    last = {}
    q, clock, day_used, day_id = q0, None, 0.0, None
    bounds = np.flatnonzero(np.diff(w)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(S)]):
        tmax = t[lo:hi].max()
        dt = 0.0 if clock is None else min(max(tmax - clock, 0), max_gap_ns) / DAY
        clock = tmax if clock is None else max(clock, tmax)
        d = int(clock // DAY)
        if d != day_id:
            day_id, day_used = d, 0.0
        votes = (Z[:, lo:hi] >= q).mean(0)
        s = S[lo:hi]
        alert = (s >= q) & (votes == 1.0)
        cand = (votes > 0) & (votes < 1)
        if rho == 0:                                   # no deferral: plain threshold
            alert = s >= q
            cand[:] = False
        n_new = 0
        idx_c = np.flatnonzero(cand)
        idx_c = idx_c[np.argsort(-votes[idx_c], kind="stable")]
        for i in idx_c:                                # serve deferrals within the share
            e = node[lo + i]
            if e in last and t[lo + i] - last[e] < dedupe_ns:
                dec[lo + i] = 2                        # part of an open incident: no new charge
                continue
            if day_used < rho * B:
                dec[lo + i] = 2
                day_used += 1
            elif s[i] >= q:
                alert[i] = True
        dec[lo:hi][alert] = 1
        for i in np.flatnonzero(dec[lo:hi] > 0) + lo:
            e = node[i]
            if e in last and t[i] - last[e] < dedupe_ns:
                continue
            last[e] = t[i]
            inc[i] = True
            n_new += 1
        n_att = int((inc[lo:hi] & y[lo:hi]).sum())
        q += eta * (n_new - n_att - B * dt)            # immediate refund (verdict delay shown separately)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return dec[inv], inc[inv]


def incident_keys(dec, inc, t, node, code):
    m = inc & (dec == code)
    return set(zip(node[m].tolist(), (t[m] // HOUR).tolist()))


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def run_policy(st, idx, cal_all, B, rho, n_val_days):
    Z = st["Z"][idx]
    S = Z.mean(0)
    cal = np.nanmean(st["Z_val"][idx], 0)
    cal = cal[np.isfinite(cal)]
    k = min(max(int(B * n_val_days), 1), len(cal))
    q0 = float(np.sort(cal)[-k])
    eta = 0.02 * (float(np.std(cal)) or 1.0)
    dec, inc = replay_defer(S, Z, st["t1"], st["w"], st["y"], st["node"], B, q0, eta, rho)
    return dec, inc


def summarize(st, dec, inc, B, tz):
    y, node = st["y"], st["node"]
    days = day_of(st["t1"], tz)
    nd = max(len(np.unique(days)), 1)
    return {
        "auto_benign_per_day": float(((dec == 1) & inc & ~y).sum() / nd),
        "defer_per_day": float(((dec == 2) & inc).sum() / nd),
        "total_benign_ratio": float((inc & ~y).sum() / nd / B),
        "attack_entities_auto": len(set(node[(dec == 1) & y])),
        "attack_entities_deferred_only": len(set(node[(dec == 2) & y]) - set(node[(dec == 1) & y])),
        "attack_entities": len(set(node[y])),
    }


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
    labels, _ = data.labels_for(members[0], args.artifact_dir)
    st = ensemble_stream(members, labels, cache_dir=cache)
    tz = labels.get("timezone", "US/Eastern")
    nvd = max(len(members[0].get("val_dates", [])), 1)
    M = st["Z"].shape[0]
    half_a, half_b = np.arange(0, M // 2), np.arange(M // 2, 2 * (M // 2))
    B = args.budget
    out = {}
    print(f"== {args.system}/{args.dataset} M={M} B={B}")

    # single-run reproducibility: pairs (0,1), (2,3), ...
    js = []
    for i in range(0, 2 * (M // 2), 2):
        da, ia = run_policy(st, [i], None, B, 0.0, nvd)
        db, ib = run_policy(st, [i + 1], None, B, 0.0, nvd)
        js.append(jaccard(incident_keys(da, ia, st["t1"], st["node"], 1),
                          incident_keys(db, ib, st["t1"], st["node"], 1)))
    out["single_run_jaccard"] = js
    print(f"  single runs           auto-alert Jaccard mean={np.mean(js):.3f} {np.round(js, 3).tolist()}")

    for rho in [0.0, 0.1, 0.25, 0.5]:
        da, ia = run_policy(st, half_a, None, B, rho, nvd)
        db, ib = run_policy(st, half_b, None, B, rho, nvd)
        j = jaccard(incident_keys(da, ia, st["t1"], st["node"], 1),
                    incident_keys(db, ib, st["t1"], st["node"], 1))
        sa, sb = summarize(st, da, ia, B, tz), summarize(st, db, ib, B, tz)
        out[f"rho={rho}"] = {"jaccard_auto": j, "A": sa, "B": sb}
        print(f"  ensemble(5) rho={rho:<4}  auto-alert Jaccard={j:.3f} | A: auto_benign/day={sa['auto_benign_per_day']:.1f} "
              f"defer/day={sa['defer_per_day']:.1f} total={sa['total_benign_ratio']:.2f}xB "
              f"attack auto={sa['attack_entities_auto']} +deferred={sa['attack_entities_deferred_only']}"
              f"/{sa['attack_entities']} | B: auto={sb['attack_entities_auto']} +def={sb['attack_entities_deferred_only']}",
              flush=True)
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    with open(os.path.join(rep, f"defer_{args.system}_{args.dataset}_{args.group}_B{int(B)}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
