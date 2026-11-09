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


def replay_budget(S, t, w, y, node, B, q0, eta, delay_ns=0, dedupe_ns=HOUR, fake=None,
                  max_gap_ns=HOUR):
    """max_gap_ns: elapsed time credited for budget is capped per batch, so gaps in the
    dataset (e.g. CADETS_E3 test days 04-06 then 04-11) do not release days of unused
    budget at once. A deployment streams continuously; dataset gaps are artifacts."""
    order = np.lexsort((t, w))
    S, t, w, y, node = S[order], t[order], w[order], y[order], node[order]
    fk = fake[order] if fake is not None else np.zeros(len(S), bool)
    flags = np.zeros(len(S), bool)
    new_inc = np.zeros(len(S), bool)
    last_alert = {}                       # entity -> time of its incident start
    q = q0
    refunds = deque()
    bounds = np.flatnonzero(np.diff(w)) + 1
    clock = None
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(S)]):
        tmax = t[lo:hi].max()
        dt = 0.0 if clock is None else min(max(tmax - clock, 0), max_gap_ns) / DAY
        clock = tmax if clock is None else max(clock, tmax)
        while refunds and refunds[0][0] <= clock:
            q -= eta * refunds.popleft()[1]
        f = S[lo:hi] >= q
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
        q += eta * (n_new - B * dt)
        if n_new_attack:
            refunds.append((clock + delay_ns, n_new_attack))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
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
        f, inc = replay_budget(stx["S"], stx["t1"], stx["w"], stx["y"], stx["node"], B,
                               warm_start(B, cold), kw.pop("eta", 0.02 * sd), fake=fake, **kw)
        real = ~fake if fake is not None else np.ones(len(f), bool)
        s = summarize(f[real], inc[real], stx["y"][real], stx["node"][real], days[real], B)
        res[tag] = s
        print(f"  {tag:38s} benign incidents/day={s['benign_incidents_per_day']:7.1f} "
              f"({s['ratio_to_budget']:4.2f}x B, worst day {s['worst_day_ratio']:4.2f}x) "
              f"attack entities hit {s['attack_entities_hit']}/{s['attack_entities']}")

    print(f"== {args.system}/{args.dataset} units={len(st['S'])}")
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
