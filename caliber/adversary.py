"""Adversarial evaluation of the decision layer (score level).

Policies, all at the same budget B incidents/day:
  static      fixed threshold = the validation-day threshold that gives B/day (eta = 0)
  caliber     incident-budget controller (caliber.budget) with entity-hour aggregation
  nodedupe    controller without aggregation (every alerting unit is an incident)

Scenarios (attacker knows the policy, controls activity, cannot alter detector/verdicts):
  S1 flood      benign-but-alerting events before the attack: volume/h x distinct entities x
                duration x lead time (gap between flood end and attack start) x score level
                (just above the starting threshold, or extreme)
  S2 drift      "boiling frog": fresh entities at a rate ramping linearly from 0 to R/h over
                D days before the attack
  S3 timing     the attacker chooses when to strike: the attack's units are shifted by an offset
                (wrapping within the test period); we report the worst / median offset
  S4 suppress   mimicry-style evasion: attack scores lowered by d validation-score SDs

Reported per run: benign incidents/day vs B (overall, worst day), attacker-generated incidents
per day (analyst load the attack adds), attack entities detected.

  python -m caliber.adversary --artifact_dir A --system velox --dataset CADETS_E3 [--scenario S1]
"""

import argparse
import json
import os

import numpy as np

from caliber import data
from caliber.budget import DAY, HOUR, replay_budget
from caliber.online import day_of, ensemble_stream

POLICIES = ["static", "caliber", "nodedupe"]


def warm_start(cal, B, n_val_days):
    k = min(max(int(B * n_val_days), 1), len(cal))
    return float(np.sort(cal)[-k])


def batch_of(st, tt):
    order = np.argsort(st["t1"])
    pos = np.clip(np.searchsorted(st["t1"][order], tt), 0, len(order) - 1)
    return st["w"][order][pos]


def add_fake(st, tt, ent, score):
    out = dict(st)
    n = len(tt)
    out["S"] = np.r_[st["S"], np.broadcast_to(score, n)]
    out["t1"] = np.r_[st["t1"], tt]
    out["w"] = np.r_[st["w"], batch_of(st, tt)]
    out["y"] = np.r_[st["y"], np.zeros(n, bool)]
    out["node"] = np.r_[st["node"], ent]
    fake = np.r_[np.zeros(len(st["S"]), bool), np.ones(n, bool)]
    return out, fake


def attack_starts(st, labels):
    return [a["start_ns"] for a in labels["attacks"]]


def flood_events(starts, rate_h, entities, hours, lead_h, rng):
    ts, es = [], []
    for s in starts:
        n = int(rate_h * hours)
        end = s - lead_h * HOUR
        ts.append(np.sort(rng.integers(end - hours * HOUR, end, n)))
        es.append(-1 - rng.integers(0, entities, n) - 10**9 * len(ts))   # distinct per attack
    return np.concatenate(ts), np.concatenate(es)


def drift_events(starts, max_rate_h, days, rng):
    """Fresh entities each event; rate ramps linearly from 0 to max_rate_h over `days`."""
    ts, es, nxt = [], [], 0
    for s in starts:
        hours = int(days * 24)
        for h in range(hours):
            rate = max_rate_h * (h + 1) / hours
            n = rng.poisson(rate)
            t0 = s - (hours - h) * HOUR
            ts.append(rng.integers(t0, t0 + HOUR, n))
            es.append(-(10**10) - np.arange(nxt, nxt + n))
            nxt += n
    return np.concatenate(ts), np.concatenate(es)


def evaluate(st, labels, cal, sd, B, policy, fake=None, tz="US/Eastern", n_val_days=1):
    q0 = warm_start(cal, B, n_val_days)
    eta = 0.0 if policy == "static" else 0.02 * sd
    dedupe = 0 if policy == "nodedupe" else HOUR
    f, inc = replay_budget(st["S"], st["t1"], st["w"], st["y"], st["node"], B, q0, eta,
                           dedupe_ns=dedupe, fake=fake)
    days = day_of(st["t1"], tz)
    real = ~fake if fake is not None else np.ones(len(f), bool)
    y, node = st["y"], st["node"]
    ben = inc & real & ~y
    per_day = {d: int((ben & (days == d)).sum()) for d in np.unique(days[real])}
    fp = np.array(list(per_day.values()), float)
    n_days = max(len(per_day), 1)
    att_nodes = set(node[real & y])
    return {
        "benign_ratio": float(fp.mean() / B), "worst_day_ratio": float(fp.max() / B),
        "attacker_incidents_per_day": float((inc & ~real).sum() / n_days),
        "attack_entities_hit": len(set(node[f & real & y])), "attack_entities": len(att_nodes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", default="da39a3ee5e")
    ap.add_argument("--budget", type=float, default=50)
    ap.add_argument("--scenario", default="all")
    args = ap.parse_args()
    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    members = data.group_manifests(data.load_manifests(
        args.artifact_dir, args.system, args.dataset, args.group))[(args.system, args.dataset, args.group)]
    labels, _ = data.labels_for(members[0], args.artifact_dir)
    st = ensemble_stream(members, labels, cache_dir=cache)
    tz = labels.get("timezone", "US/Eastern")
    cal = st["S_val"][np.isfinite(st["S_val"])]
    sd = float(np.std(cal)) or 1.0
    nvd = max(len(members[0].get("val_dates", [])), 1)
    B = args.budget
    q0 = warm_start(cal, B, nvd)
    starts = attack_starts(st, labels)
    rng = np.random.default_rng(0)
    rows = []

    def run(scn, params, stx=st, fake=None):
        for pol in POLICIES:
            r = evaluate(stx, labels, cal, sd, B, pol, fake, tz, nvd)
            r.update({"scenario": scn, "policy": pol, **params})
            rows.append(r)
            print(f"  {scn:9s} {json.dumps(params):70s} {pol:8s} benign={r['benign_ratio']:5.2f}xB "
                  f"(worst {r['worst_day_ratio']:5.2f}) attacker_inc/day={r['attacker_incidents_per_day']:8.1f} "
                  f"hit={r['attack_entities_hit']}/{r['attack_entities']}", flush=True)

    print(f"== {args.system}/{args.dataset} B={B} units={len(st['S'])}")
    run("none", {})
    if args.scenario in ("all", "S1"):
        hi, lo = float(np.max(cal)) * 2 + 1, q0 + 0.05 * sd
        grid = [(r, e, 6, 0, "extreme") for r in [100, 1000, 10000] for e in [1, 10, 100, 1000]]
        grid += [(1000, 100, h, 0, "extreme") for h in [1, 24]]
        grid += [(1000, 100, 6, lead, "extreme") for lead in [6, 24]]
        grid += [(1000, e, 6, 0, "low") for e in [10, 100]]
        for rate, ent, hours, lead, lvl in grid:
            tt, ee = flood_events(starts, rate, ent, hours, lead, rng)
            stf, fake = add_fake(st, tt, ee, hi if lvl == "extreme" else lo)
            run("S1-flood", {"rate_h": rate, "entities": ent, "hours": hours, "lead_h": lead,
                             "level": lvl}, stf, fake)
    if args.scenario in ("all", "S2"):
        hi = float(np.max(cal)) * 2 + 1
        for days in [1, 3]:
            for rmax in [10, 100]:
                tt, ee = drift_events(starts, rmax, days, rng)
                stf, fake = add_fake(st, tt, ee, hi)
                run("S2-drift", {"days": days, "max_new_entities_h": rmax}, stf, fake)
    if args.scenario in ("all", "S3"):
        t_lo, t_hi = st["t1"].min(), st["t1"].max()
        span = t_hi - t_lo
        y = st["y"]
        for off_h in range(0, int(span / HOUR), 6):
            stx = dict(st)
            t = st["t1"].copy()
            t[y] = t_lo + (t[y] - t_lo + off_h * HOUR) % span
            stx["t1"] = t
            w = st["w"].copy()
            w[y] = batch_of({"t1": st["t1"][~y], "w": st["w"][~y]}, t[y])
            stx["w"] = w
            run("S3-timing", {"offset_h": off_h}, stx)
    if args.scenario in ("all", "S4"):
        for d in [0.5, 1, 2, 4]:
            stx = dict(st)
            S = st["S"].copy()
            S[st["y"]] -= d * sd
            stx["S"] = S
            run("S4-suppress", {"minus_sd": d}, stx)
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    with open(os.path.join(rep, f"adversary_{args.system}_{args.dataset}_{args.group}_B{int(B)}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
