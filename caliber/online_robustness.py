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
                   delay_ns=0, miss_p=0.0, fake=None, seed=0, ceil_factor=None):
    """Batch-wise online control with delayed / noisy feedback and optional fake units.

    method:
      static   fixed conformal threshold
      aci      adaptive conformal inference on alpha_t (needs verdicts)
      qtrack   quantile tracking driven by analyst verdicts (err = dismissed alert)
      rate     quantile tracking driven by ALL alerts immediately (label-free alert-rate
               control); a verdict that an alert was a real attack later REFUNDS its
               increment, so attacks do not consume the false-alarm budget
    ceil_factor: if set, the threshold may never exceed the static conformal threshold at
          alpha / ceil_factor. Worst case under any feedback manipulation is then a static
          detector at that stricter level.
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
    q_cap = np.inf
    if ceil_factor:
        q_cap = conformal.threshold(cal, alpha / ceil_factor)
        if not np.isfinite(q_cap):
            q_cap = np.max(cal)
    pending = deque()            # (release_time, n_false_alarms, n_units) or ("refund", n_tp)
    bounds = np.flatnonzero(np.diff(w)) + 1
    clock = -np.inf               # monotone clock: PIDSMaker batches overlap in time
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(S)]):
        clock = max(clock, t[lo:hi].max())
        while pending and pending[0][0] <= clock:        # apply verdicts that have arrived
            item = pending.popleft()
            if item[1] == "refund":                      # rate mode: confirmed attacks
                q_t -= eta * item[2]
                continue
            _, n_fa, n = item
            # n unit-level updates, n_fa of them with err=1 (additive, so summed)
            a_t += gamma * (alpha * n - n_fa)
            q_t += eta * (n_fa - alpha * n)
        q_t = min(q_t, q_cap)
        if method in ("qtrack", "rate"):
            thr = q_t
        elif method == "aci":
            thr = conformal.threshold(cal, min(a_t, 1.0)) if a_t > 0 else np.inf
        else:
            thr = conformal.threshold(cal, alpha)
        f = S[lo:hi] >= thr
        flags[lo:hi] = f
        benign_alert = f & (~y[lo:hi] | fk[lo:hi])
        if method == "rate":
            # immediate, label-free update: every alert counts as a false alarm now
            n_alert = int(f.sum())
            n = hi - lo
            q_t += eta * (n_alert * (1 - alpha) - (n - n_alert) * alpha)
            q_t = min(q_t, q_cap)
            # later, analysts confirm the real attacks -> refund those increments
            n_tp = int((f & ~benign_alert).sum())
            n_tp += rng.binomial(int(benign_alert.sum()), miss_p)   # analyst error
            if n_tp:
                pending.append((clock + delay_ns, "refund", n_tp * (1 - alpha)))
            continue
        # analyst error: some true false alarms are (wrongly) marked malicious
        n_fa = int(benign_alert.sum() - rng.binomial(int(benign_alert.sum()), miss_p))
        pending.append((clock + delay_ns, n_fa, hi - lo))
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
    for a in [1e-4, 1e-3]:
        show(f"alpha={a} static", run(st, labels, cal, days, alpha=a, method="static"), a)
    variants = [("qtrack", None), ("rate", None), ("qtrack", 10), ("rate", 10)]
    high = float(np.max(cal)) * 2 + 1
    for m, c in variants:
        tag = f"{m}{'+ceil' + str(c) if c else ''}"
        kw = dict(method=m, ceil_factor=c)
        for a in [1e-4, 3e-4, 1e-3]:
            show(f"alpha={a} {tag}", run(st, labels, cal, days, alpha=a, **kw), a)
        a = 1e-3
        for d in [1, 4, 12, 24]:
            show(f"alpha={a} {tag} delay={d}h",
                 run(st, labels, cal, days, alpha=a, delay_ns=d * HOUR, **kw), a)
        for p in [0.1, 0.3]:
            show(f"alpha={a} {tag} miss_p={p}", run(st, labels, cal, days, alpha=a, miss_p=p, **kw), a)
        for e in [0.01, 0.2]:
            show(f"alpha={a} {tag} eta={e}", run(st, labels, cal, days, alpha=a, eta_factor=e, **kw), a)
        for n, h in [(100, 6), (1000, 6)]:
            stf, fake = inject_flood(st, labels, n, h, high)
            s = run(stf, labels, cal, day_of(stf["t1"], labels.get("timezone", "US/Eastern")),
                    alpha=a, fake=fake, **kw)
            show(f"alpha={a} {tag} flood={n}/h x{h}h", s, a)
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    with open(os.path.join(rep, f"online_robust_{args.system}_{args.dataset}_{args.group}.json"),
              "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
