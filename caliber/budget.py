"""Incident-budget control: the operator sets B alerts (incidents) per day.

Why a per-day incident budget instead of a per-unit false-alarm rate:
  * analysts investigate entities, not entity-window units; repeated alerts on the
    same entity within an hour are one investigation (standard alert aggregation);
  * SOC tolerance is stated per host per day (Dong et al., CCS'23);
  * deduplication bounds alert flooding: k attacker-controlled entities can add at
    most k incidents per hour, however many events they generate.

Controller (P-control on the threshold, Angelopoulos et al. 2023, time-based):
    q <- q + eta * (I_batch - B * dt_days)
where I_batch counts NEW incidents (entity first alerted in the current hour).
Every incident counts immediately (label-free: attacks are rare, so incidents are
almost all false alarms); when an analyst later confirms an incident as a real
attack, its increment is refunded, so attacks do not consume the budget. Long-run
average incidents/day -> B for any score sequence (bounded threshold movement).

Reported: incidents/day vs B (overall, worst day), attack incidents, attack
entities detected (entity-level recall, as in PIDSMaker node evaluation).

  python -m caliber.budget --artifact_dir A --system velox --dataset CADETS_E3
"""

import argparse
import json
import os
from collections import deque

import numpy as np

from caliber import data
from caliber.online import day_of, ensemble_stream

HOUR = 3600 * 10**9
DAY = 24 * HOUR


def calibrate_rise_rate(S_val, t_val, B, window_h=6, factor=2.0):
    """Fastest benign drift of a B/day threshold, per hour, measured on the validation day.

    For sliding windows of `window_h` hours (stride 1 h) we take the threshold that would give
    B/day within that window; the rate is `factor` x the largest hour-to-hour change. Benign
    behavior never required the threshold to move faster than this, so a controller limited to
    this rate still tracks benign shift but caps how fast an alert flood can push it up."""
    t0 = t_val.min()
    hours = int((t_val.max() - t0) // HOUR) + 1
    k = max(int(round(B * window_h / 24)), 1)
    qs = []
    for h in range(0, max(hours - window_h, 0) + 1):
        m = (t_val >= t0 + h * HOUR) & (t_val < t0 + (h + window_h) * HOUR)
        if m.sum() > k:
            qs.append(np.sort(S_val[m])[-k])
    if len(qs) < 2:
        return None
    return factor * float(np.max(np.abs(np.diff(qs))))


def replay_budget(S, t, w, y, node, B, q0, eta, delay_ns=0, dedupe_ns=HOUR, fake=None,
                  max_gap_ns=HOUR, eta_scale=None, window_units=50000, spread_stat="tail",
                  rise_rate_per_hour=None, std_window=None, std_init=None, return_q=False):
    """Incident-budget controller replay. Returns (alert flags, new-incident flags), plus the
    threshold in force for each unit if return_q.

    max_gap_ns: elapsed time credited for budget is capped per batch, so gaps in the dataset
        (e.g. CADETS_E3 test days 04-06 then 04-11) do not release days of unused budget.
    eta_scale / spread_stat: scale-free step, eta_t = eta_scale * spread of the most recent
        `window_units` scores; "tail" = 99.9th pct - median, "mad" = 1.4826 x MAD.
    rise_rate_per_hour: upward threshold moves limited to this many score units per hour of
        credited time (downward moves unlimited). See calibrate_rise_rate.
    std_window / std_init: online robust standardization. Each batch's scores are standardized
        by the median and 1.4826*MAD of the previous `std_window` units (std_init = (median,
        mad) of the validation scores until enough history exists), and the threshold q lives
        in standardized units. A shift of the bulk of the score distribution is absorbed
        immediately, while a minority of attacker-generated scores (an alert flood) cannot
        move a median or a MAD (50% breakdown point)."""
    order = np.lexsort((t, w))
    S, t, w, y, node = S[order], t[order], w[order], y[order], node[order]
    fk = fake[order] if fake is not None else np.zeros(len(S), bool)
    flags = np.zeros(len(S), bool)
    new_inc = np.zeros(len(S), bool)
    q_at = np.zeros(len(S))
    last_alert = {}                       # entity -> time of its incident start
    q = q0
    refunds = deque()
    recent, n_recent = deque(), 0
    hist, n_hist = deque(), 0
    loc, scale = std_init if std_init is not None else (0.0, 1.0)
    allowance = rise_rate_per_hour if rise_rate_per_hour is not None else None
    bounds = np.flatnonzero(np.diff(w)) + 1
    clock = None
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(S)]):
        tmax = t[lo:hi].max()
        dt = 0.0 if clock is None else min(max(tmax - clock, 0), max_gap_ns) / DAY
        clock = tmax if clock is None else max(clock, tmax)
        if allowance is not None:
            allowance = min(allowance + rise_rate_per_hour * dt * 24, rise_rate_per_hour)
        if eta_scale is not None:
            recent.append(S[lo:hi])
            n_recent += hi - lo
            while n_recent - len(recent[0]) >= window_units:
                n_recent -= len(recent.popleft())
            r = np.concatenate(recent)
            if spread_stat == "mad":
                med = np.median(r)
                spread = 1.4826 * float(np.median(np.abs(r - med)))
            else:
                spread = float(np.quantile(r, 0.999) - np.median(r))
            eta = eta_scale * (spread if spread > 0 else 1.0)
        while refunds and refunds[0][0] <= clock:
            q -= eta * refunds.popleft()[1]
        s_b = S[lo:hi]
        if std_window is not None:
            if n_hist >= min(std_window, 10000):
                h = np.concatenate(hist)
                loc = float(np.median(h))
                mad = 1.4826 * float(np.median(np.abs(h - loc)))
                scale = mad if mad > 0 else scale
            s_eval = (s_b - loc) / scale
            hist.append(s_b)
            n_hist += hi - lo
            while n_hist - len(hist[0]) >= std_window:
                n_hist -= len(hist.popleft())
        else:
            s_eval = s_b
        q_at[lo:hi] = q
        f = s_eval >= q
        flags[lo:hi] = f
        n_new, n_new_attack = 0, 0
        for i in np.flatnonzero(f) + lo:
            e = node[i]
            if e in last_alert and t[i] - last_alert[e] < dedupe_ns:
                continue
            last_alert[e] = t[i]
            new_inc[i] = True
            n_new += 1
            if y[i] and not fk[i]:
                n_new_attack += 1
        step = eta * (n_new - B * dt)
        if allowance is not None and step > 0:
            step = min(step, allowance)
            allowance -= step
        q += step
        if n_new_attack:
            refunds.append((clock + delay_ns, n_new_attack))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    if return_q:
        return flags[inv], new_inc[inv], q_at[inv]
    return flags[inv], new_inc[inv]


def summarize(flags, new_inc, y, node, days, B):
    per_day = {}
    for d in np.unique(days):
        m = days == d
        per_day[d] = int((new_inc & m & ~y).sum())
    fp_per_day = np.array(list(per_day.values()), float)
    att_nodes = set(node[y])
    hit_nodes = set(node[flags & y])
    return {"benign_incidents_per_day": float(fp_per_day.mean()),
            "ratio_to_budget": float(fp_per_day.mean() / B),
            "worst_day_ratio": float(fp_per_day.max() / B),
            "attack_incidents": int((new_inc & y).sum()),
            "attack_entities_hit": len(hit_nodes), "attack_entities": len(att_nodes),
            "per_day": per_day}


def flood(st, labels, rate_per_h, hours, n_entities, score, seed=0):
    """Injected benign-but-alerting events before each attack, from n_entities fake entities."""
    rng = np.random.default_rng(seed)
    ts, es = [], []
    for a in labels["attacks"]:
        n = int(rate_per_h * hours)
        ts.append(np.sort(rng.integers(a["start_ns"] - hours * HOUR, a["start_ns"], n)))
        es.append(-1 - rng.integers(0, n_entities, n))          # negative ids: fake entities
    tt, ee = np.concatenate(ts), np.concatenate(es)
    # assign each fake event to the batch of the nearest real unit in time
    order = np.argsort(st["t1"])
    pos = np.clip(np.searchsorted(st["t1"][order], tt), 0, len(order) - 1)
    ww = st["w"][order][pos]
    out = {k: v for k, v in st.items()}
    out["S"] = np.r_[st["S"], np.full(len(tt), score)]
    out["t1"] = np.r_[st["t1"], tt]
    out["w"] = np.r_[st["w"], ww]
    out["y"] = np.r_[st["y"], np.zeros(len(tt), bool)]
    out["node"] = np.r_[st["node"], ee]
    fake = np.r_[np.zeros(len(st["S"]), bool), np.ones(len(tt), bool)]
    return out, fake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", default="da39a3ee5e")
    ap.add_argument("--only_adaptive", action="store_true",
                    help="compare fixed vs scale-free (adaptive) step only")
    ap.add_argument("--only_std", action="store_true",
                    help="compare raw vs online robust standardization only")
    args = ap.parse_args()
    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    members = data.group_manifests(data.load_manifests(
        args.artifact_dir, args.system, args.dataset, args.group))[(args.system, args.dataset, args.group)]
    labels, _ = data.labels_for(members[0], args.artifact_dir)
    st = ensemble_stream(members, labels, cache_dir=cache)
    tz = labels.get("timezone", "US/Eastern")
    cal = st["S_val"][np.isfinite(st["S_val"])]
    sd = float(np.std(cal)) or 1.0
    n_val_days = max(len(members[0].get("val_dates", [])), 1)
    high = float(np.max(cal)) * 2 + 1
    res = {}

    def warm_start(B, cold=False):
        """Threshold that would have produced ~B alerts/day on the validation day(s).
        Ignores dedupe (alerts on validation are mostly distinct entities)."""
        if cold:
            return float(np.quantile(cal, 0.999))
        k = min(max(int(B * n_val_days), 1), len(cal))
        return float(np.sort(cal)[-k])

    def go(tag, B, stx=st, fake=None, cold=False, **kw):
        days = day_of(stx["t1"], tz)
        q0 = kw.pop("q0", None)
        q0 = warm_start(B, cold) if q0 is None else q0
        f, inc = replay_budget(stx["S"], stx["t1"], stx["w"], stx["y"], stx["node"], B,
                               q0, kw.pop("eta", 0.02 * sd), fake=fake, **kw)
        real = ~fake if fake is not None else np.ones(len(f), bool)
        s = summarize(f[real], inc[real], stx["y"][real], stx["node"][real], days[real], B)
        res[tag] = s
        print(f"  {tag:38s} benign incidents/day={s['benign_incidents_per_day']:7.1f} "
              f"({s['ratio_to_budget']:4.2f}x B, worst day {s['worst_day_ratio']:4.2f}x) "
              f"attack entities hit {s['attack_entities_hit']}/{s['attack_entities']}")

    print(f"== {args.system}/{args.dataset} units={len(st['S'])}")
    if args.only_std:
        med_v = float(np.median(cal))
        mad_v = 1.4826 * float(np.median(np.abs(cal - med_v))) or 1.0
        cal_std = (cal - med_v) / mad_v
        sd_std = float(np.std(cal_std)) or 1.0

        def q0_std(B):
            k = min(max(int(B * n_val_days), 1), len(cal_std))
            return float(np.sort(cal_std)[-k])

        for B in [10, 50, 200]:
            go(f"B={B} raw", B)
            for win in [50000, 200000]:
                go(f"B={B} std win={win}", B, q0=q0_std(B), eta=0.02 * sd_std,
                   std_window=win, std_init=(med_v, mad_v))
        for rate, k in [(1000, 10), (1000, 100)]:
            stf, fake = flood(st, labels, rate, 6, k, high)
            go(f"B=50 raw flood {rate}/h x6h from {k}", 50, stx=stf, fake=fake)
            go(f"B=50 std flood {rate}/h x6h from {k}", 50, stx=stf, fake=fake, q0=q0_std(50),
               eta=0.02 * sd_std, std_window=200000, std_init=(med_v, mad_v))
        return
    if args.only_adaptive:
        for B in [10, 50, 200]:
            go(f"B={B} fixed-eta", B)
            for c in [0.1, 0.5, 2.0]:
                go(f"B={B} adaptive-MAD c={c}", B, eta_scale=c, spread_stat="mad")
        for c in [0.5, 2.0]:
            for rate, k in [(1000, 10), (1000, 100)]:
                stf, fake = flood(st, labels, rate, 6, k, high)
                go(f"B=50 MAD c={c} flood {rate}/h x6h from {k}", 50, stx=stf, fake=fake,
                   eta_scale=c, spread_stat="mad")
        return
    for B in [10, 50, 200]:
        go(f"B={B}", B)
    go("B=50 cold-start", 50, cold=True)
    go("B=50 no-gap-cap", 50, max_gap_ns=10**18)
    for d in [1, 24]:
        go(f"B=50 delay={d}h", 50, delay_ns=d * HOUR)
    for e in [0.005, 0.1]:
        go(f"B=50 eta={e}sd", 50, eta=e * sd)
    go("B=50 no-dedupe", 50, dedupe_ns=0)
    for rate, k in [(1000, 1), (1000, 10), (1000, 100), (100, 100)]:
        stf, fake = flood(st, labels, rate, 6, k, high)
        go(f"B=50 flood {rate}/h x6h from {k} entities", 50, stx=stf, fake=fake)
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    with open(os.path.join(rep, f"budget_{args.system}_{args.dataset}_{args.group}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
