"""Stress tests for online false-alarm control (see caliber/online.py).

The basic replay assumes every alert is labelled immediately and correctly. Here we
relax that, one factor at a time, for quantile tracking (qtrack) and ACI:

  alpha     realistic budgets: 1e-4 .. 1e-2 per entity-window unit
  delay     analyst verdicts arrive `delay_h` hours after the alert
  noise     analyst marks a true false alarm as malicious with prob `miss_p`
            (so it is not counted as a false alarm), the dominant real-world error
  eta       step size of qtrack, as a multiple of the calibration-score std
  flood     an adversary injects `flood_n` benign-but-alerting units per hour during
            the `flood_h` hours before each attack starts; analysts dismiss them, which
            pushes the threshold up. We measure recall on the real attack units.

All runs report FAR on real benign units, worst-day FAR, and TP / recall.

  python -m caliber.online_robustness --artifact_dir A --system velox --dataset CADETS_E3 --group G
"""

import argparse
import json
import os
from collections import deque

import numpy as np

from caliber import conformal, data
from caliber.online import day_of, ensemble_stream, summarize

HOUR = 3600 * 10**9


def replay_general(S, t, w, y, cal, alpha, method="qtrack", eta_factor=0.05, gamma=0.005,
                   delay_ns=0, miss_p=0.0, fake=None, seed=0):
    """Batch-wise online control with delayed / noisy feedback and optional fake units.

    fake: boolean mask of injected units (scored, alertable, analyst-dismissed, but
          excluded from the reported metrics)."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((t, w))
    S, t, w, y = S[order], t[order], w[order], y[order]
    fk = fake[order] if fake is not None else np.zeros(len(S), bool)
    flags = np.zeros(len(S), bool)
    a_t = alpha
    q_t = conformal.threshold(cal, alpha)
    if not np.isfinite(q_t):
        q_t = np.max(cal)
    eta = eta_factor * (np.std(cal) or 1.0)
    pending = deque()            # (release_time, n_false_alarms, n_units)
    bounds = np.flatnonzero(np.diff(w)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(S)]):
        now = t[lo:hi].max()
        while pending and pending[0][0] <= now:          # apply verdicts that have arrived
            _, n_fa, n = pending.popleft()
            for _ in range(n - n_fa):
                a_t += gamma * alpha
                q_t -= eta * alpha
            for _ in range(n_fa):
                a_t += gamma * (alpha - 1)
                q_t += eta * (1 - alpha)
        if method == "qtrack":
            thr = q_t
        elif method == "aci":
            thr = conformal.threshold(cal, min(a_t, 1.0)) if a_t > 0 else np.inf
        else:
            thr = conformal.threshold(cal, alpha)
        f = S[lo:hi] >= thr
        flags[lo:hi] = f
        benign_alert = f & (~y[lo:hi] | fk[lo:hi])
        # analyst error: some true false alarms are (wrongly) marked malicious
        n_fa = int(benign_alert.sum() - rng.binomial(int(benign_alert.sum()), miss_p))
        pending.append((now + delay_ns, n_fa, hi - lo))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return flags[inv]


def inject_flood(st, labels, flood_n, flood_h, score):
    """Append fake benign units with a high score before each attack start."""
    ts, ws = [], []
    w_of_t = lambda x: st["w"][np.argmin(np.abs(st["t1"] - x))]
    for a in labels["attacks"]:
        start = a["start_ns"]
        n = int(flood_n * flood_h)
        tt = np.linspace(start - flood_h * HOUR, start, n, endpoint=False).astype(np.int64)
        ts.append(tt)
        ws.append(np.array([w_of_t(x) for x in tt]))
    if not ts:
        return st, np.zeros(len(st["S"]), bool)
    tt, ww = np.concatenate(ts), np.concatenate(ws)
    out = dict(st)
    out["S"] = np.r_[st["S"], np.full(len(tt), score)]
    out["t1"] = np.r_[st["t1"], tt]
    out["w"] = np.r_[st["w"], ww]
    out["y"] = np.r_[st["y"], np.zeros(len(tt), bool)]
    fake = np.r_[np.zeros(len(st["S"]), bool), np.ones(len(tt), bool)]
    return out, fake


def run(st, labels, cal, days, **kw):
    fake = kw.pop("fake", None)
    f = replay_general(st["S"], st["t1"], st["w"], st["y"], cal, fake=fake, **kw)
    real = ~fake if fake is not None else np.ones(len(f), bool)
    return summarize(f[real], st["y"][real], np.asarray(days)[real])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", default="da39a3ee5e")
    args = ap.parse_args()
    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    members = data.group_manifests(data.load_manifests(
        args.artifact_dir, args.system, args.dataset, args.group))[(args.system, args.dataset, args.group)]
    labels, _ = data.load_labels(args.artifact_dir, args.dataset, members[0]["ground_truth_version"])
    st = ensemble_stream(members, labels, cache_dir=cache)
    days = day_of(st["t1"], labels.get("timezone", "US/Eastern"))
    cal = st["S_val"][np.isfinite(st["S_val"])]
    res = {}

    def show(tag, s, a):
        res[tag] = s
        print(f"  {tag:42s} FAR={s['far']:.6f} ({s['far']/a:5.2f}x) worst-day={s['worst_day_far']/a:5.2f}x "
              f"alerts={s['alerts']:6d} TP={s['tp']:4d}")

    print(f"== {args.system}/{args.dataset} units={len(st['S'])} malicious={int(st['y'].sum())}")
    for a in [1e-4, 3e-4, 1e-3, 1e-2]:
        for m in ["static", "aci", "qtrack"]:
            show(f"alpha={a} {m}", run(st, labels, cal, days, alpha=a, method=m), a)
    a = 1e-3
    for d in [1, 4, 12, 24]:
        show(f"alpha={a} qtrack delay={d}h", run(st, labels, cal, days, alpha=a, delay_ns=d * HOUR), a)
    for p in [0.1, 0.3, 0.5]:
        show(f"alpha={a} qtrack miss_p={p}", run(st, labels, cal, days, alpha=a, miss_p=p), a)
    for e in [0.01, 0.05, 0.2, 1.0]:
        show(f"alpha={a} qtrack eta={e}", run(st, labels, cal, days, alpha=a, eta_factor=e), a)
    high = float(np.max(cal)) * 2 + 1
    for n in [10, 100, 1000]:
        for h in [1, 6]:
            stf, fake = inject_flood(st, labels, n, h, high)
            s = run(stf, labels, cal, day_of(stf["t1"], labels.get("timezone", "US/Eastern")),
                    alpha=a, fake=fake)
            show(f"alpha={a} qtrack flood={n}/h x{h}h", s, a)
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    with open(os.path.join(rep, f"online_robust_{args.system}_{args.dataset}_{args.group}.json"),
              "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
