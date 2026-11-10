"""Controlled automation: three-tier decisions with operator targets.

  tier A  automate-act   S >= q_act      handled without analysts (e.g. automatic enrichment,
                                         containment, ticketing); operator sets a precision
                                         target pi for these cases
  tier B  abstain-review q_rev <= S < q_act   analyst queue; the budget controller keeps
                                         tier-B cases + audits at ~B per day (Proposition 1)
  tier C  automate-dismiss S < q_rev

The automated tiers stay controllable without attack labels through AUDIT SAMPLING: each
new tier-A case is sent to an analyst with probability a; audited verdicts give an unbiased
precision estimate, and q_act moves until the lower confidence bound of tier-A precision meets
pi. Audits count against the analyst budget, so analysts never see more than ~B cases/day.

Cases are (key, hour) with key = provenance lineage when available (else entity).

Compared against "all alerts to triage" (every case with S >= q_rev goes to an analyst):
analyst load, automation share, realized tier-A precision vs pi, and attack entities handled
(automated in A, or reviewed in B within capacity).

  python -m caliber.tiers --artifact_dir A --system velox --dataset CADETS_E3
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import beta

from caliber import data
from caliber.budget import DAY, HOUR
from caliber.lineage import lineage_keys, load_parents, process_predicate
from caliber.online import day_of, ensemble_stream


def wilson_lower(pos, n, conf=0.9):
    if n == 0:
        return 0.0
    return float(beta.ppf(1 - conf, pos + 0.5, n - pos + 0.5))      # Jeffreys interval


def replay_tiers(S, t, w, y, keys, B, q0, eta, pi, audit, eta_act, q_act0, rng,
                 max_gap_ns=HOUR, window_audits=50, B_auto=None, S_act=None):
    """B_auto: if set, q_act is driven by a false-automation budget instead of a precision
    target: at most ~B_auto benign cases per day may be acted on automatically. Same
    label-free controller as the review budget (Proposition 1): every new tier-A case counts
    immediately and confirmed attacks are refunded. No audit sample is needed for control.
    S_act: statistic for the act tier (default S). With the confidence-interval rule, S is the
    upper bound (a unit is dismissed only if it is confidently below q_rev) and S_act the
    lower bound (a unit is automated only if it is confidently above q_act)."""
    S_act = S if S_act is None else S_act
    order = np.lexsort((t, w))
    case_of = np.full(len(S), -1, np.int64)        # case index per alerted unit
    cases = []                                     # dicts: key, hour, tier, audited
    open_case = {}
    q_rev, q_act, clock = q0, max(q_act0, q0), None
    audits = []                                    # recent audited tier-A verdicts
    bounds = np.flatnonzero(np.diff(w[order])) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(order)]):
        ix = order[lo:hi]
        tmax = t[ix].max()
        dt = 0.0 if clock is None else min(max(tmax - clock, 0), max_gap_ns) / DAY
        clock = tmax if clock is None else max(clock, tmax)
        n_rev, n_att_rev, n_auto, n_att_auto = 0, 0, 0, 0
        for i in ix[S[ix] >= q_rev]:
            kh = (keys[i], int(t[i] // HOUR))
            c = open_case.get(kh)
            if c is None:
                tier = "A" if S_act[i] >= q_act else "B"
                aud = tier == "A" and rng.random() < audit
                cases.append({"tier": tier, "audited": aud, "y": bool(y[i])})
                c = len(cases) - 1
                open_case[kh] = c
                if tier == "B" or aud:
                    n_rev += 1
                    n_att_rev += int(y[i])
                if tier == "A":
                    n_auto += 1
                    n_att_auto += int(y[i])
                if aud:
                    audits.append(bool(y[i]))
                    audits[:] = audits[-window_audits:]
            else:
                cases[c]["y"] = cases[c]["y"] or bool(y[i])
            case_of[i] = c
        # review budget (tier B + audits); confirmed attacks refunded
        q_rev += eta * (n_rev - n_att_rev - B * dt)
        if B_auto is not None:
            # false-automation budget: same controller as the review budget
            q_act += eta_act * (n_auto - n_att_auto - B_auto * dt)
        # automation threshold: meet the precision target on audited tier-A cases
        elif len(audits) >= 10:
            lb = wilson_lower(sum(audits), len(audits))
            if lb < pi:
                q_act += eta_act
            elif lb > pi + 0.1:
                q_act -= 0.25 * eta_act
        q_act = max(q_act, q_rev)
    return case_of, cases


def evaluate(st, case_of, cases, B, tz, attacks=None):
    """attacks: label-export attack list; an attack counts as handled when any of its nodes is
    in an automated case or in a case reviewed within capacity (attack-level detection, as in
    Orthrus and the ADP metric of Bilot et al.)."""
    y, node, t = st["y"], st["node"], st["t1"]
    days = day_of(t, tz)
    nd = max(len(np.unique(days)), 1)
    tierA = np.array([c["tier"] == "A" for c in cases])
    ylab = np.array([c["y"] for c in cases])
    aud = np.array([c["audited"] for c in cases])
    nA, nB = int(tierA.sum()), int((~tierA).sum())
    # attack entities: automated (in a tier-A case) or reviewed (tier-B case within capacity)
    att_units = np.flatnonzero((case_of >= 0) & y)
    auto_ent = {node[i] for i in att_units if tierA[case_of[i]]}
    # analyst capacity: first B tier-B/audit cases per day (FIFO by opening time)
    first_t = {}
    for i in np.flatnonzero(case_of >= 0):
        c = case_of[i]
        first_t[c] = min(first_t.get(c, t[i]), t[i])
    rev_cases = [c for c in range(len(cases)) if (not tierA[c]) or aud[c]]
    by_day = {}
    for c in rev_cases:
        d = day_of(np.array([first_t[c]]), tz)[0]
        by_day.setdefault(d, []).append(c)
    reviewed = set()
    for d, cs in by_day.items():
        reviewed |= set(sorted(cs, key=lambda c: first_t[c])[: int(B)])
    rev_ent = {node[i] for i in att_units if case_of[i] in reviewed}
    # per day (convergence of both controllers): benign review cases, false automations
    per_day = {}
    for c, tc in first_t.items():
        d = str(day_of(np.array([tc]), tz)[0])
        row = per_day.setdefault(d, [0, 0])
        if not ylab[c]:
            row[0 if not tierA[c] else 1] += 1
    return {
        "per_day_benign_review_and_false_auto": dict(sorted(per_day.items())),
        "cases_A_per_day": nA / nd, "cases_B_per_day": nB / nd, "audits_per_day": float(aud.sum()) / nd,
        "analyst_load_per_day": (nB + float(aud.sum())) / nd,
        "all_to_triage_load_per_day": (nA + nB) / nd,
        "automation_share": nA / max(nA + nB, 1),
        "precision_A": float(ylab[tierA].mean()) if nA else float("nan"),
        "precision_B": float(ylab[~tierA].mean()) if nB else float("nan"),
        "attack_entities_automated": len(auto_ent),
        "attack_entities_reviewed": len(rev_ent - auto_ent),
        "attack_entities_handled": len(auto_ent | rev_ent),
        "attack_entities": len(set(node[y])),
        **_attack_level(attacks, set(node[y]), auto_ent | rev_ent, auto_ent),
    }


def _attack_level(attacks, present, handled, automated):
    if not attacks:
        return {}
    in_test = [set(a["nids"]) & present for a in attacks]
    in_test = [a for a in in_test if a]
    return {"attacks": len(in_test),
            "attacks_handled": sum(bool(a & handled) for a in in_test),
            "attacks_automated": sum(bool(a & automated) for a in in_test)}


def _with_borrower(labels, manifest, artifact_dir):
    """Identity-borrowing runs re-attribute attack events to a benign process (code injection).
    That process is compromised, so it joins every attack's ground truth; unit labels already
    restrict credit to the attack's time window."""
    argv = manifest.get("argv", [])
    if "--adversary" not in argv:
        return labels
    spec = json.loads(argv[argv.index("--adversary") + 1])
    if not spec.get("borrow"):
        return labels
    import glob
    for p in glob.glob(os.path.join(artifact_dir, "construction", "construction", "*",
                                    manifest["dataset"], "adversary.json")):
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("spec") == spec and rec.get("borrower") is not None:
            b = int(rec["borrower"])
            labels = dict(labels, attacks=[dict(a, nids=sorted(set(a["nids"]) | {b}))
                                           for a in labels["attacks"]])
            labels["malicious_nids"] = sorted(set(labels["malicious_nids"]) | {b})
            print(f"  borrower {b} added to the ground truth")
            return labels
    print("  WARNING: borrow run without a recorded borrower")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", default="da39a3ee5e")
    ap.add_argument("--budget", type=float, default=50)
    ap.add_argument("--kappa", type=float, default=1.645,
                    help="confidence-interval half-width in ensemble standard errors")
    ap.add_argument("--rules", default="ensemble,ensemble_upper,ensemble_ci,single_run")
    ap.add_argument("--quick", action="store_true",
                    help="only all-to-triage and B_auto=0.25 (eta_act=1.0)")
    ap.add_argument("--single_all", action="store_true",
                    help="single-run rules for every member (mean and range), not only member 0")
    args = ap.parse_args()
    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    members = data.group_manifests(data.load_manifests(
        args.artifact_dir, args.system, args.dataset, args.group))[(args.system, args.dataset, args.group)]
    labels, meta = data.labels_for(members[0], args.artifact_dir)
    labels = _with_borrower(labels, members[0], args.artifact_dir)
    st = ensemble_stream(members, labels, cache_dir=cache)
    tz = labels.get("timezone", "US/Eastern")
    B = args.budget
    parents = load_parents(args.artifact_dir, args.dataset)
    lin = (lineage_keys(st["node"], st["partner"], parents, process_predicate(parents, meta), 1)
           if parents is not None else st["node"].astype(np.int64))
    nvd = max(len(members[0].get("val_dates", [])), 1)
    out = {}
    print(f"== {args.system}/{args.dataset} B={B} lineage={'yes' if parents is not None else 'no'}")
    M = st["Z"].shape[0]
    sem = np.nan_to_num(np.nanstd(st["Z"], 0) / np.sqrt(M))
    sem_val = np.nan_to_num(np.nanstd(st["Z_val"], 0) / np.sqrt(M))
    rules = []
    for r_ in args.rules.split(","):    # single-run rules: every member, reported as mean/range
        rules += [f"{r_}#{m}" for m in range(M)] if r_.startswith("single_run") and args.single_all else [r_]
    for score_name in rules:
        S_act = cal_act = None
        name, _, idx = score_name.partition("#")
        static = name.endswith("_static")    # fixed validation threshold, no controller
        base = name[:-len("_static")] if static else name
        if base == "single_run":
            S, cal = st["Z"][int(idx or 0)], st["Z_val"][int(idx or 0)]
        elif base == "ensemble":
            S, cal = st["S"], st["S_val"]
        elif base == "ensemble_upper":   # review on the upper bound, act on the mean
            S, cal = st["S"] + args.kappa * sem, st["S_val"] + args.kappa * sem_val
            S_act, cal_act = st["S"], st["S_val"]
        else:   # ensemble_ci: review on the upper bound, act on the lower bound
            S, cal = st["S"] + args.kappa * sem, st["S_val"] + args.kappa * sem_val
            S_act, cal_act = st["S"] - args.kappa * sem, st["S_val"] - args.kappa * sem_val
        cal = cal[np.isfinite(cal)]
        cal_act = cal if cal_act is None else cal_act[np.isfinite(cal_act)]
        k = min(max(int(B * nvd), 1), len(cal))
        q0 = float(np.sort(cal)[-k])
        sd = float(np.std(cal)) or 1.0
        # baseline: all alerts to triage (no automation); then false-automation budgets.
        # Warm start: the validation threshold that would give ~B_auto/day (top-1 when < 1/day).
        # The act tier's budget is ~100x smaller than the review budget, so its controller needs
        # a proportionally larger step; we sweep it (in validation-score SDs).
        settings = [(None, None)] + [(b, e) for b in [0.25, 1.0] for e in [0.2, 1.0]]
        if args.quick:
            settings = [(None, None), (0.25, 1.0)]
        if static:
            settings = [(None, None)]
        for b_auto, eta_a in settings:
            for audit in [0.05]:
                rng = np.random.default_rng(0)
                ka = max(int((b_auto or 1) * nvd), 1)
                q_act0 = float(np.sort(cal_act)[-ka])
                case_of, cases = replay_tiers(
                    S, st["t1"], st["w"], st["y"], lin, B, q0, 0.0 if static else 0.02 * sd,
                    2.0, audit if b_auto is not None else 0.0,
                    (eta_a or 0.02) * sd, q_act0 if b_auto is not None else np.inf, rng,
                    B_auto=b_auto, S_act=S_act)
                r = evaluate(st, case_of, cases, B, tz, labels.get("attacks"))
                fa = [c for c in cases if c["tier"] == "A" and not c["y"]]
                nd = max(len(np.unique(day_of(st["t1"], tz))), 1)
                r["false_automations_per_day"] = len(fa) / nd
                tag = f"{score_name} " + ("all-to-triage" if b_auto is None
                                          else f"B_auto={b_auto} eta_act={eta_a}")
                out[tag] = r
                if "#" in score_name:
                    continue                      # summarized over members below
                print(f"  {tag:34s} false-auto/day={r.get('false_automations_per_day', 0):5.2f} analyst/day={r['analyst_load_per_day']:6.1f} "
                      f"(all-to-triage {r['all_to_triage_load_per_day']:6.1f}) auto share={r['automation_share']:.2f} "
                      f"prec A={r['precision_A']:.2f} B={r['precision_B']:.3f} | attack entities "
                      f"auto={r['attack_entities_automated']} reviewed={r['attack_entities_reviewed']} "
                      f"handled={r['attack_entities_handled']}/{r['attack_entities']} | attacks "
                      f"{r.get('attacks_handled', '-')}/{r.get('attacks', '-')} "
                      f"(auto {r.get('attacks_automated', '-')})", flush=True)
    # single-run rules over all members: mean [min-max]
    groups = {}
    for tag, r in out.items():
        name, sep, rest = tag.partition("#")
        if sep:
            groups.setdefault(name + " " + rest.split(" ", 1)[1], []).append(r)
    keys = ["analyst_load_per_day", "false_automations_per_day", "attack_entities_handled",
            "attacks_handled", "attacks_automated"]
    for tag, rs in groups.items():
        summ = {k: [float(r.get(k, np.nan)) for r in rs] for k in keys}
        out[tag + " (members)"] = {k: {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)),
                                       "max": float(np.nanmax(v))} for k, v in summ.items()}
        fmt = lambda k: f"{np.nanmean(summ[k]):.1f} [{np.nanmin(summ[k]):.0f}-{np.nanmax(summ[k]):.0f}]"
        print(f"  {tag:34s} (mean [min-max] over {len(rs)} members) analyst/day={fmt('analyst_load_per_day')} "
              f"false-auto/day={np.nanmean(summ['false_automations_per_day']):.2f} | attack entities "
              f"handled={fmt('attack_entities_handled')}/{rs[0]['attack_entities']} | attacks "
              f"{fmt('attacks_handled')}/{rs[0].get('attacks', '-')} (auto {fmt('attacks_automated')})",
              flush=True)
    rep = os.path.join(args.artifact_dir, "caliber", "reports")
    with open(os.path.join(rep, f"tiers_{args.system}_{args.dataset}_{args.group}_B{int(B)}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
