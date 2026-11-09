"""Is provenance-lineage aggregation a viable flood defense? Feasibility study.

  (A) compression   benign alerts at budget B: distinct (entity,hour) vs (lineage,hour)
  (B) separation    do attack lineages coincide with busy benign lineages (which would hide
                    the attack inside an already-open benign incident)?
  (C) budget        controller with incidents counted per (key, hour), key = entity or lineage-k
  (D) floods        E flood entities spread over L independent footholds (one lineage each);
                    compare entity- vs lineage-aggregation, and analysts reviewing distinct
                    lineages first (each lineage at most once per day) within capacity B/day

  python -m caliber.lineage_study --artifact_dir A --system velox --dataset CADETS_E3
"""

import argparse
import json
import os

import numpy as np

from caliber import data
from caliber.adversary import add_fake, flood_events, warm_start
from caliber.budget import HOUR, replay_budget
from caliber.lineage import lineage_keys, load_parents, process_predicate
from caliber.online import day_of, ensemble_stream


def run(st, keys, B, q0, eta, fake=None):
    return replay_budget(st["S"], st["t1"], st["w"], st["y"], keys, B, q0, eta, fake=fake)


def metrics(st, f, inc, keys, B, tz, fake=None, lineage_first=False):
    days = day_of(st["t1"], tz)
    real = ~fake if fake is not None else np.ones(len(f), bool)
    y, node, t = st["y"], st["node"], st["t1"]
    nd = max(len(np.unique(days[real])), 1)
    hour = t // HOUR
    reviewed_keys = set()
    for d in np.unique(days):
        idx = np.flatnonzero(inc & (days == d))
        idx = idx[np.argsort(t[idx], kind="stable")]
        if lineage_first:                      # one incident per lineage per day first
            seen, first, rest = set(), [], []
            for i in idx:
                (rest if keys[i] in seen else first).append(i)
                seen.add(keys[i])
            idx = np.array(first + rest, dtype=np.int64)
        for i in idx[: int(B)]:
            reviewed_keys.add((keys[i], hour[i]))
    # an attack entity is seen if one of its alerting units falls in a reviewed (key, hour)
    att = np.flatnonzero(f & real & y)
    seen_att = {node[i] for i in att if (keys[i], hour[i]) in reviewed_keys}
    return {"benign_incidents_per_day": float((inc & real & ~y).sum() / nd),
            "attacker_incidents_per_day": float((inc & ~real).sum() / nd),
            "attack_entities_hit": len(set(node[att])),
            "attack_entities_reviewed": len(seen_att),
            "attack_entities": len(set(node[real & y]))}


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
    labels, meta = data.labels_for(members[0], args.artifact_dir)
    parents = load_parents(args.artifact_dir, args.dataset)
    if parents is None:
        print(f"no lineage for {args.dataset}")
        return
    st = ensemble_stream(members, labels, cache_dir=cache)
    tz = labels.get("timezone", "US/Eastern")
    cal = st["S_val"][np.isfinite(st["S_val"])]
    sd = float(np.std(cal)) or 1.0
    nvd = max(len(members[0].get("val_dates", [])), 1)
    B = args.budget
    q0, eta = warm_start(cal, B, nvd), 0.02 * sd
    is_proc = process_predicate(parents, meta)
    keys = {"entity": st["node"].astype(np.int64)}
    for k in [1, 2, 3]:
        keys[f"lineage{k}"] = lineage_keys(st["node"], st["partner"], parents, is_proc, k)
    res = {}
    print(f"== {args.system}/{args.dataset} B={B} units={len(st['S'])} "
          f"processes in tree={len(parents)}")

    # (A)+(B) on the entity-level controller's alerts
    f, inc = run(st, keys["entity"], B, q0, eta)
    y, hour = st["y"], st["t1"] // HOUR
    ben = f & ~y
    for name, kk in keys.items():
        n_ent = len(set(zip(st["node"][ben].tolist(), hour[ben].tolist())))
        n_key = len(set(zip(kk[ben].tolist(), hour[ben].tolist())))
        att_keys = set(kk[y].tolist())
        busy = {}
        for x in kk[ben].tolist():
            busy[x] = busy.get(x, 0) + 1
        shared = sum(1 for a in att_keys if busy.get(a, 0) > 0)
        res[f"A/B {name}"] = {"benign_entity_hours": n_ent, "benign_key_hours": n_key,
                              "attack_keys": len(att_keys), "attack_keys_shared_with_benign_alerts": shared}
        print(f"  [{name:9s}] benign alert (entity,h)={n_ent:6d} -> (key,h)={n_key:6d} "
              f"(x{n_ent / max(n_key, 1):.1f}) | attack keys={len(att_keys)}, shared with benign alerts={shared}")

    # (C) budget control with key aggregation
    for name in ["entity", "lineage1", "lineage2"]:
        f, inc = run(st, keys[name], B, q0, eta)
        m = metrics(st, f, inc, keys[name], B, tz)
        res[f"C {name}"] = m
        print(f"  [budget {name:9s}] benign incidents/day={m['benign_incidents_per_day']:6.1f} "
              f"hit={m['attack_entities_hit']}/{m['attack_entities']} reviewed@B={m['attack_entities_reviewed']}")

    # (D) floods: E entities over L footholds, extreme scores, 6h before each attack
    rng = np.random.default_rng(0)
    high = float(np.max(cal)) * 2 + 1
    starts = [a["start_ns"] for a in labels["attacks"]]
    for E, L in [(10, 1), (100, 1), (1000, 1), (100, 10), (1000, 10), (1000, 100)]:
        tt, ee = flood_events(starts, 1000, E, 6, 0, rng)
        lin = -(5 * 10**11) - (np.abs(ee) % L)            # each fake entity -> one of L footholds
        stf, fake = add_fake(st, tt, ee, high)
        for name in ["entity", "lineage1"]:
            kf = np.r_[keys[name], ee if name == "entity" else lin]
            f, inc = run(stf, kf, B, q0, eta, fake)
            for lf in ([False, True] if name == "lineage1" else [False]):
                m = metrics(stf, f, inc, kf, B, tz, fake, lineage_first=lf)
                tag = f"D E={E} L={L} {name}{'+lineage-first review' if lf else ''}"
                res[tag] = m
                print(f"  [flood E={E:4d} L={L:3d} {name:9s}{' +lin-first' if lf else '           '}] "
                      f"attacker inc/day={m['attacker_incidents_per_day']:7.1f} hit={m['attack_entities_hit']}"
                      f"/{m['attack_entities']} reviewed@B={m['attack_entities_reviewed']} "
                      f"benign inc/day={m['benign_incidents_per_day']:.1f}", flush=True)
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    with open(os.path.join(rep, f"lineage_{args.system}_{args.dataset}_{args.group}_B{int(B)}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
