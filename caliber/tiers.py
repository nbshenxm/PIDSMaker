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
from caliber.lineage import case_keys, executable_keys
from caliber.online import day_of, ensemble_stream

# Review-controller step: a full day's budget error (B surplus cases) moves the threshold by
# ETA_DAY validation standard deviations, whatever B is. A fixed per-case step makes the
# controller sluggish at small budgets and jumpy at large ones; Proposition 1 holds for any step.
ETA_DAY = 2.0
# review on the ensemble upper bound U = S + KAPPA * sem (one-sided 95%)
KAPPA = 1.645


def review_step(cal, B):
    return ETA_DAY * (float(np.std(cal)) or 1.0) / max(B, 1.0)


def wilson_lower(pos, n, conf=0.9):
    if n == 0:
        return 0.0
    return float(beta.ppf(1 - conf, pos + 0.5, n - pos + 0.5))      # Jeffreys interval


def case_threshold(S_val, t_val, keys_val, B, n_val_days):
    """Warm start: the validation threshold at which ~B cases per day would have opened, with
    cases counted as the controller counts them ((case key, hour) pairs). Calibrating on alert
    units instead starts far too high once units are grouped into cases: one case holds many
    units (about 18 on ATLASv2, 30 on OpTC), so a unit-calibrated start delivers a few percent of
    B and the controller needs many days to come down."""
    S_val = np.asarray(S_val, float)
    m = np.isfinite(S_val)
    s = S_val[m]
    if len(s) == 0:
        return np.inf
    order = np.argsort(-s, kind="stable")
    kk = np.asarray(keys_val, np.int64)[m][order]
    hh = (np.asarray(t_val, np.int64)[m][order] // HOUR).astype(np.int64)
    _, first = np.unique(np.stack([kk, hh], 1), axis=0, return_index=True)
    first = np.sort(first)                       # positions (in score order) of each new case
    target = max(int(B * n_val_days), 1)
    pos = first[min(target, len(first)) - 1]
    return float(s[order][pos])


def flood_cap_for(B):
    """Default per-batch cap on new individual cases: one hour of the review budget."""
    return max(1, int(np.ceil(B / 24.0)))


# Token bucket (rate limiting, as in network traffic policing and IDS alert rate filters): an
# analyst-facing case needs a token; tokens refill at B per day up to a depth of BUCKET_H hours
# of budget. Over any window of t hours analysts receive at most D + B*t/24 cases, D = B*BUCKET_H/24,
# whatever the data or an attacker does. Default: one shift (8 h).
BUCKET_H = 8.0


def bucket_depth(B, bucket_h=BUCKET_H):
    return max(1.0, B * bucket_h / 24.0)


def replay_tiers(S, t, w, y, keys, B, q0, eta, pi, audit, eta_act, q_act0, rng,
                 max_gap_ns=HOUR, window_audits=50, B_auto=None, S_act=None,
                 flood_cap="auto", refund_delay_ns=0, act_key=None, act_seen=None,
                 bucket_h=None, skip_keys=None, act_rate=None, release_h=None):
    """B_auto: if set, q_act is driven by a false-automation budget instead of a precision
    target: at most ~B_auto benign cases per day may be acted on automatically. Same
    label-free controller as the review budget (Proposition 1): every new tier-A case counts
    immediately and confirmed attacks are refunded. No audit sample is needed for control.
    S_act: statistic for the act tier (default S). With the confidence-interval rule, S is the
    upper bound (a unit is dismissed only if it is confidently below q_rev) and S_act the
    lower bound (a unit is automated only if it is confidently above q_act).
    flood_cap: at most this many new individual cases per batch ("auto": one hour of budget,
    None: no cap). Alerting units of a batch open cases in descending score order; once the cap
    is reached, the batch's remaining new alerts join one bulk review case, so a single batch
    delivers at most flood_cap + 1 cases to analysts. After a flood the threshold rises just
    enough for the batch to have stayed within the cap, by at most eta*B (one day of budget).
    Bulk cases are never automated.
    refund_delay_ns: an analyst's verdict arrives this long after a case opens; a case that is
    malicious by then (any of its units so far) is refunded at that time, never earlier.
    act_key / act_seen: novelty gate for the act tier. A case may be automated only if the
    executable of its first unit (act_key) has not alerted before: not in act_seen (alerts on the
    benign validation days) and not in any earlier case. Benign false alarms recur (updates,
    scheduled jobs, the same noisy tool); attack processes are new.
    bucket_h: token-bucket admission instead of the per-batch flood cap (flood_cap is ignored).
    A new review case (or audit) needs a token; tokens refill at B/day over real elapsed time up
    to D = bucket_depth(B, bucket_h). Without a token, the alert joins one pending summary case
    that is delivered, as a single item, when the next token arrives (never automated). The
    threshold controller counts demand (every case that would have opened) and each batch's
    update is clipped to +-eta*D (integrator anti-windup): a burst, or a gap in the data, moves the
    threshold by at most one bucket's worth, which replaces the gap cap and the flood step.
    skip_keys: (case key, hour) pairs whose case another controller already delivered (online
    round-robin, caliber.multiview); their alerts open no case here and are not charged.
    act_rate: token bucket of depth one action for the act tier, refilling at act_rate actions per
    day. Every automated action needs a token, whatever its verdict, so at most 1 + act_rate * t
    actions are taken in any t days (attacker-triggered ones included) and no batch can act more
    than once beyond the controller's threshold; a unit that finds no token goes to review.
    release_h (with bucket_h): new review cases are held until their release period of release_h
    hours closes (the case aggregation period), then admitted in descending score order while
    tokens last; the rest join one summary item delivered with the next token. Without it, cases
    are admitted in arrival order. The bound is the same; the delay is at most release_h."""
    seen = set(act_seen or ())
    act_tokens = 1.0
    hold, hold_slot, over = [], None, []           # release_h: held cases; overflow awaiting a token
    REL = None if release_h is None else release_h * HOUR
    S_act = S if S_act is None else S_act
    use_bucket = bucket_h is not None
    D = bucket_depth(B, bucket_h) if use_bucket else None
    tokens = D
    pend = None                                    # pending summary case awaiting a token
    cap = None if use_bucket else (flood_cap_for(B) if flood_cap == "auto" else flood_cap)
    order = np.lexsort((t, w))
    case_of = np.full(len(S), -1, np.int64)        # case index per alerted unit
    cases = []                                     # dicts: tier, audited, y, flood
    open_case = {}
    pending = []                                   # (due time, case) awaiting a verdict
    q_rev, q_act, clock = q0, max(q_act0, q0), None
    audits = []                                    # recent audited tier-A verdicts
    bounds = np.flatnonzero(np.diff(w[order])) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(order)]):
        ix = order[lo:hi]
        tmax = t[ix].max()
        elapsed = 0.0 if clock is None else max(tmax - clock, 0) / DAY
        dt = elapsed if use_bucket else min(elapsed * DAY, max_gap_ns) / DAY
        clock = tmax if clock is None else max(clock, tmax)
        if use_bucket:
            tokens = min(D, tokens + B * elapsed)
        if act_rate is not None:
            act_tokens = min(1.0, act_tokens + act_rate * elapsed)
        if REL is not None and use_bucket:
            slot = int(tmax // REL)
            if hold and hold_slot is not None and slot != hold_slot:
                tokens = _release(hold, over, cases, tokens, (hold_slot + 1) * REL)
                hold = []
            if over and tokens >= 1.0:             # overflow: one summary item with the next token
                for c in over:
                    cases[c]["t_deliver"] = tmax
                tokens -= 1.0
                over = []
            hold_slot = slot
        n_rev, n_auto, opened, bulk, n_bulk = 0, 0, 0, None, 0
        n_over = 0                                 # bucket: review cases that found no token
        al = ix[S[ix] >= q_rev]
        for i in al[np.argsort(-S[al], kind="stable")]:
            kh = (keys[i], int(t[i] // HOUR))
            c = open_case.get(kh)
            if c is None and skip_keys is not None and kh in skip_keys:
                continue                           # case already delivered by another view
            if c is None and use_bucket and REL is None and tokens < 1.0:
                wants_review = not (S_act[i] >= q_act and (act_key is None or act_key[i] not in seen)
                                    and (act_rate is None or act_tokens >= 1.0))
                if wants_review:                   # no token: join the pending summary case
                    if pend is None:
                        cases.append({"tier": "B", "audited": False, "y": False, "flood": True,
                                      "t_deliver": None})
                        pend = len(cases) - 1
                    n_over += 1
                    c = open_case[kh] = pend
            if c is None and cap is not None and opened >= cap:
                if bulk is None:               # flood: one bulk review case for the rest
                    cases.append({"tier": "B", "audited": False, "y": False, "flood": True})
                    bulk = len(cases) - 1
                    n_rev += 1
                    pending.append((t[i] + refund_delay_ns, bulk))
                if n_bulk == 0:
                    s_bulk = S[i]              # highest score that overflowed the cap
                n_bulk += 1                    # cases the flood would have opened
                c = open_case[kh] = bulk
            if act_key is not None:
                novel = act_key[i] not in seen
                seen.add(act_key[i])
            else:
                novel = True
            if c is None:
                tier = "A" if (S_act[i] >= q_act and novel
                               and (act_rate is None or act_tokens >= 1.0)) else "B"
                if tier == "A" and act_rate is not None:
                    act_tokens -= 1.0
                aud = tier == "A" and rng.random() < audit
                cases.append({"tier": tier, "audited": aud, "y": False, "flood": False})
                c = len(cases) - 1
                open_case[kh] = c
                opened += 1
                n_rev += int(tier == "B" or aud)
                n_auto += int(tier == "A")
                if use_bucket and REL is not None and tier == "B":
                    cases[c]["t_deliver"] = None   # held until its release period closes
                    hold.append((float(S[i]), c))
                elif use_bucket and (tier == "B" or aud):
                    tokens -= 1.0
                pending.append((t[i] + refund_delay_ns, c))
                if aud:
                    audits.append(bool(y[i]))
                    audits[:] = audits[-window_audits:]
            cases[c]["y"] = cases[c]["y"] or bool(y[i])
            case_of[i] = c
        if use_bucket and pend is not None and tokens >= 1.0:
            # deliver the summary of the overflow as one item with the next token
            cases[pend]["t_deliver"] = tmax
            tokens -= 1.0
            pending.append((tmax + refund_delay_ns, pend))
            pend = None
        # verdicts that have arrived: confirmed attacks are refunded
        ref_rev = ref_auto = 0
        if pending:
            due = [p for p in pending if p[0] <= clock]
            pending = [p for p in pending if p[0] > clock]
            for _, c in due:
                if cases[c]["y"]:
                    if cases[c]["tier"] == "A":
                        ref_auto += 1
                        ref_rev += int(cases[c]["audited"])
                    else:
                        ref_rev += 1
        # review budget (tier B + audits), counting the cases actually delivered
        q_prev = q_rev
        if use_bucket:
            # demand (every review case that would have opened) vs budget, clipped to one bucket
            step = eta * (n_rev + n_over - ref_rev - B * dt)
            q_rev += float(np.clip(step, -eta * D, eta * D))
        else:
            q_rev += eta * (n_rev - ref_rev - B * dt)
        if bulk is not None:
            # flood: raise the threshold just enough that this batch would have stayed within the
            # cap (just above its highest overflowing score), by at most one day of budget (eta*B).
            # A dense flood just above the threshold moves it little; a flood far above it moves it
            # at most one day's worth, so a flood cannot blind the controller for long. The step only
            # ever raises q relative to the plain update, so Proposition 1 holds with N = cap + 1.
            # (Also bounding the step by the flood's size slowed the climb under a large shift:
            # CLEARSCOPE at 2.9x B. This rule never exceeds B; its cost is some unused budget.)
            q_rev = max(q_rev, min(q_prev + eta * B, float(s_bulk) + 1e-9))
        if B_auto is not None:
            # false-automation budget: same controller as the review budget
            dt_act = min(elapsed * DAY, max_gap_ns) / DAY
            q_act += eta_act * (n_auto - ref_auto - B_auto * dt_act)
        # automation threshold: meet the precision target on audited tier-A cases
        elif len(audits) >= 10:
            lb = wilson_lower(sum(audits), len(audits))
            if lb < pi:
                q_act += eta_act
            elif lb > pi + 0.1:
                q_act -= 0.25 * eta_act
        q_act = max(q_act, q_rev)
    if REL is not None and use_bucket and hold:    # end of stream: release the last period
        _release(hold, over, cases, tokens, (hold_slot + 1) * REL)
    return case_of, cases


def _release(hold, over, cases, tokens, t_rel):
    """Admit held review cases at t_rel in descending score order while tokens last; the rest are
    marked as summary (flood) cases and appended to `over` (delivered as one item later)."""
    for _, c in sorted(hold, key=lambda x: -x[0]):
        if tokens >= 1.0:
            tokens -= 1.0
            cases[c]["t_deliver"] = t_rel
        else:
            cases[c]["flood"] = True
            over.append(c)
    return tokens


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
    for c, cs in enumerate(cases):                 # bucket summary cases: delivered later or never
        if "t_deliver" in cs and c in first_t:
            if cs["t_deliver"] is None:
                del first_t[c]
            else:
                first_t[c] = cs["t_deliver"]
    rev_cases = [c for c in range(len(cases)) if ((not tierA[c]) or aud[c]) and c in first_t]
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
        "flood_cases_per_day": sum(bool(c.get("flood")) for c in cases) / nd,
        "precision_A": float(ylab[tierA].mean()) if nA else float("nan"),
        "precision_B": float(ylab[~tierA].mean()) if nB else float("nan"),
        "attack_entities_automated": len(auto_ent),
        "attack_entities_reviewed": len(rev_ent - auto_ent),
        "attack_entities_handled": len(auto_ent | rev_ent),
        "attack_entities": len(set(node[y])),
        **_attack_level(attacks, node, t, np.flatnonzero(y),
                        [i for i in att_units if tierA[case_of[i]] or case_of[i] in reviewed],
                        [i for i in att_units if tierA[case_of[i]]]),
    }


def unit_attacks(node, t, idx, attacks, slack_ns=HOUR):
    """For malicious units idx: the attacks each one belongs to. A unit belongs to attack k when
    its entity is in k's ground truth AND its time lies in k's interval (window end t within
    [start, end + slack]). Entities shared by several attacks (e.g. a browser used in two THEIA
    attacks) are credited only to the attack running at the time."""
    from caliber.pilot import LABEL_PAD_NS
    sets = [set(a["nids"]) for a in attacks]
    out = {}
    for i in idx:
        out[i] = [k for k, a in enumerate(attacks) if node[i] in sets[k]
                  and a["start_ns"] - LABEL_PAD_NS <= t[i] <= a["end_ns"] + slack_ns]
    return out


def _attack_level(attacks, node, t, mal_units, handled_units, automated_units):
    if not attacks:
        return {}
    ua = unit_attacks(node, t, mal_units, attacks)
    present = {k for i in mal_units for k in ua[i]}
    handled = {k for i in handled_units for k in ua.get(i, [])}
    automated = {k for i in automated_units for k in ua.get(i, [])}
    return {"attacks": len(present), "attacks_handled": len(handled & present),
            "attacks_automated": len(automated & present)}


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
    for p in glob.glob(os.path.join(artifact_dir, "construction", manifest["dataset"],
                                    "construction", "*", "adversary.json")):
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
    ap.add_argument("--act_rate", type=float, default=1.0,
                    help="act-tier token bucket: actions per day, depth one (<= 0: no bucket)")
    ap.add_argument("--no_act_cap", action="store_true",
                    help="do not saturate the act score at the largest benign validation score")
    ap.add_argument("--quick", action="store_true",
                    help="only all-to-triage and B_auto=0.25 (eta_act=1.0)")
    ap.add_argument("--single_all", action="store_true",
                    help="single-run rules for every member (mean and range), not only member 0")
    ap.add_argument("--scorer", default="ensemble", choices=["ensemble", "abstain", "plain3"],
                    help="unit scores: ensemble (default) or caliber.abstain_score (S = pooled "
                         "scorer, Z = per-seed scorer, so the interval is over training runs)")
    ap.add_argument("--seeds", default="0,1,2", help="seeds for --scorer abstain/plain3")
    ap.add_argument("--tag", default="", help="suffix for the report file")
    ap.add_argument("--act_on", default="score", choices=["score", "plain"],
                    help="abstain scorers: act tier on the scorer (default) or on the plain-decoder mean b")
    args = ap.parse_args()
    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    members = data.group_manifests(data.load_manifests(
        args.artifact_dir, args.system, args.dataset, args.group))[(args.system, args.dataset, args.group)]
    labels, meta = data.labels_for(members[0], args.artifact_dir)
    labels = _with_borrower(labels, members[0], args.artifact_dir)
    if args.scorer == "ensemble":
        st = ensemble_stream(members, labels, cache_dir=cache)
    else:
        from caliber import abstain_score
        st = abstain_score.scored_stream(args.scorer, args.artifact_dir, args.system, args.dataset,
                                         labels, seeds=tuple(int(s) for s in args.seeds.split(",")))
    tz = labels.get("timezone", "US/Eastern")
    B = args.budget
    lin, has_lin = case_keys(st, args.artifact_dir, args.dataset, meta)
    lin_val, _ = case_keys(st, args.artifact_dir, args.dataset, meta, split="val")
    exe_key, exe_val = executable_keys(st, args.artifact_dir, args.dataset, meta)
    nvd = max(len(members[0].get("val_dates", [])), 1)
    out = {}
    print(f"== {args.system}/{args.dataset} B={B} lineage={'yes' if has_lin else 'no'}")
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
        elif base.startswith("ensemble_upper"):
            # review on the upper bound; act on the mean (default), or on the median member
            # ("_median": a majority of runs must clear the act threshold); "_novel" also requires
            # that the acting executable has not alerted before (validation or earlier in the run)
            S, cal = st["S"] + args.kappa * sem, st["S_val"] + args.kappa * sem_val
            if "_median" in base:
                S_act, cal_act = np.nanmedian(st["Z"], 0), np.nanmedian(st["Z_val"], 0)
            else:
                S_act, cal_act = st["S"], st["S_val"]
            if args.act_on == "plain" and "views" in st:   # act on the plain decoder's mean
                S_act, cal_act = st["views"]["b"][1], st["views"]["b"][0]
        else:   # ensemble_ci: review on the upper bound, act on the lower bound
            S, cal = st["S"] + args.kappa * sem, st["S_val"] + args.kappa * sem_val
            S_act, cal_act = st["S"] - args.kappa * sem, st["S_val"] - args.kappa * sem_val
        q0 = case_threshold(cal, st["t_val"], lin_val, B, nvd)   # B cases/day on validation
        cal = cal[np.isfinite(cal)]
        cal_act = cal if cal_act is None else cal_act[np.isfinite(cal_act)]
        # The act score saturates at the largest benign validation score: a unit more extreme than
        # every benign validation unit is equally certain. This fixes s_max in Prop. 1, and the warm
        # start below sits at or just under it, so the additive constant stays below one case.
        if not args.no_act_cap:
            S_act = np.minimum(S if S_act is None else S_act, float(np.max(cal_act)))
        sd = float(np.std(cal)) or 1.0
        # executables that already alerted on the benign validation day(s) at the start threshold
        U_val = st["S_val"] + args.kappa * sem_val
        seen_val = set(exe_val[np.isfinite(U_val) & (U_val >= q0)].tolist())
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
                    S, st["t1"], st["w"], st["y"], lin, B, q0, 0.0 if static else review_step(cal, B),
                    2.0, audit if b_auto is not None else 0.0,
                    (eta_a or 0.02) * sd, q_act0 if b_auto is not None else np.inf, rng,
                    B_auto=b_auto, S_act=S_act, flood_cap=None,
                    bucket_h=None if static else BUCKET_H,
                    act_key=exe_key if "_novel" in base else None,
                    act_seen=seen_val if "_novel" in base else None,
                    act_rate=args.act_rate if args.act_rate > 0 else None)
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
    sc = "" if args.scorer == "ensemble" else f"_{args.scorer}"
    with open(os.path.join(rep, f"tiers_{args.system}_{args.dataset}_{args.group}{sc}{args.tag}_B{int(B)}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
