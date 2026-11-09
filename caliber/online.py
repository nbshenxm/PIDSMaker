"""Replay test days in time order and compare false-alarm control strategies.

Motivation (pilot, 2026-11-09): a threshold calibrated on one benign validation day
misses its nominal FAR on later days by 0.2x-50x, because days are not
exchangeable. With only a handful of benign days, no distribution-free method can
certify a small alpha across days, so we evaluate ONLINE control that needs no
exchangeability and only uses feedback a SOC already produces: the analyst's
verdict on each raised alert (missed attacks are never labelled).

Strategies (all on the ensemble score S of entity-window units, in time order):
  static     threshold = conformal (1-alpha) quantile of the validation-day scores
  aci        adaptive conformal inference (Gibbs & Candes 2021): alpha_t += gamma*(alpha - err_t),
             threshold_t = (1-alpha_t) quantile of the validation scores
  qtrack     quantile tracking (Angelopoulos et al. 2023, P-control): q_t += eta*(err_t - alpha),
             i.e. the threshold itself moves, so it can leave the calibration range

err_t = 1 if unit t raised an alert that the analyst marks benign, else 0
(an alert on a truly malicious unit is not a false alarm; unalerted units are 0).
Feedback is applied after each PIDSMaker batch, i.e. decisions within a batch share
one threshold, as they would in a deployment that processes windows.

Usage:
  python -m caliber.online --artifact_dir A [--system velox] [--dataset CADETS_E3] [--group G]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from caliber import conformal, data
from caliber.pilot import normaliser, unit_labels, unit_table

ALPHAS = [1e-3, 1e-2]


def ensemble_stream(members, labels, window="batch", epoch_rule="min_val_loss", cache_dir=None):
    """Ensemble score per entity-window unit (val + test), with times and labels."""
    zs_val, zs_test, idx_test, tt = [], [], None, None
    for m in members:
        ep = data.select_epoch(m, epoch_rule, cache_dir)
        val = unit_table(data.read_split(m["edge_losses_dir"], "val", ep, cache_dir), "entity_window", window)
        test = unit_table(data.read_split(m["edge_losses_dir"], "test", ep, cache_dir), "entity_window", window)
        if idx_test is None:
            idx_test, tt = test.index, test
            val_idx, tv = val.index, val
        test = test.reindex(idx_test)
        val = val.reindex(val_idx)
        f = normaliser(val["score"].values, "robust_z")
        zs_val.append(f(val["score"].values))
        zs_test.append(f(test["score"].values))
    y = unit_labels(tt, "entity_window", labels)
    return {
        "S_val": np.nanmean(np.vstack(zs_val), 0),
        "S": np.vstack(zs_test).mean(0),
        "t1": tt["t1"].values, "w": idx_test.get_level_values("w").values, "y": y,
        "node": idx_test.get_level_values("node").values,
        "Z": np.vstack(zs_test),                   # per-member scores (for vote splits)
        "Z_val": np.vstack(zs_val),
        "t_val": tv["t1"].values,                  # validation unit times (rate calibration)
    }


def replay(S, w, y, cal, alpha, method, gamma=0.005, eta=None):
    """Run one strategy over units grouped by batch w (sorted by time). Returns flags."""
    order = np.argsort(w, kind="stable")
    S, w, y = S[order], w[order], y[order]
    flags = np.zeros(len(S), bool)
    cal_sorted = np.sort(cal)
    a_t = alpha
    q_t = conformal.threshold(cal, alpha)
    if not np.isfinite(q_t):
        q_t = cal_sorted[-1]
    eta = eta if eta is not None else 0.05 * (np.std(cal) or 1.0)
    bounds = np.flatnonzero(np.diff(w)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(S)]):
        if method == "static":
            thr = conformal.threshold(cal, alpha)
        elif method == "aci":
            thr = conformal.threshold(cal, min(max(a_t, 0.0), 1.0)) if a_t > 0 else np.inf
        elif method == "qtrack":
            thr = q_t
        else:
            raise ValueError(method)
        f = S[lo:hi] >= thr
        flags[lo:hi] = f
        err = (f & ~y[lo:hi]).astype(float)           # analyst-confirmed false alarms
        # one update per unit in the batch (same as unit-level ACI, applied in order)
        for e in err:
            a_t += gamma * (alpha - e)
            q_t += eta * (e - alpha)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return flags[inv]


def day_of(t_ns, tz="US/Eastern"):
    return pd.to_datetime(t_ns, unit="ns", utc=True).tz_convert(tz).strftime("%Y-%m-%d").values


def summarize(flags, y, days):
    benign = ~y
    out = {"far": float(flags[benign].mean()), "tp": int((flags & y).sum()),
           "recall": float((flags & y).sum() / max(y.sum(), 1)), "alerts": int(flags.sum())}
    per_day = {}
    for d in np.unique(days):
        m = days == d
        per_day[d] = {"far": float(flags[m & benign].mean()) if (m & benign).any() else None,
                      "alerts": int(flags[m].sum()), "tp": int((flags[m] & y[m]).sum())}
    out["per_day"] = per_day
    fars = [v["far"] for v in per_day.values() if v["far"] is not None]
    out["worst_day_far"] = float(max(fars)) if fars else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system")
    ap.add_argument("--dataset")
    ap.add_argument("--group")
    ap.add_argument("--min_members", type=int, default=5)
    args = ap.parse_args()
    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    groups = data.group_manifests(data.load_manifests(args.artifact_dir, args.system, args.dataset, args.group))
    results = {}
    for key, members in sorted(groups.items()):
        if len(members) < args.min_members:
            continue
        labels, _ = data.labels_for(members[0], args.artifact_dir)
        st = ensemble_stream(members, labels, cache_dir=cache)
        days = day_of(st["t1"], labels.get("timezone", "US/Eastern"))
        cal = st["S_val"][np.isfinite(st["S_val"])]
        res = {}
        print(f"== {key} M={len(members)} units={len(st['S'])} malicious={int(st['y'].sum())}")
        for a in ALPHAS:
            for meth in ["static", "aci", "qtrack"]:
                s = summarize(replay(st["S"], st["w"], st["y"], cal, a, meth), st["y"], days)
                res[f"{meth}@{a}"] = s
                print(f"  alpha={a:<6} {meth:7s} FAR={s['far']:.5f} ({s['far']/a:5.2f}x)  "
                      f"worst-day={s['worst_day_far']:.5f} ({s['worst_day_far']/a:5.2f}x)  "
                      f"alerts={s['alerts']:6d} TP={s['tp']:4d} recall={s['recall']:.3f}")
        results["|".join(key)] = res
    os.makedirs(rep, exist_ok=True)
    with open(os.path.join(rep, "online_replay.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
