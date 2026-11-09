"""Realizable test-time adversarial transforms of the provenance event stream (CALIBER).

Applied during graph construction to TEST dates only, so a frozen detector (--from_weights)
scores the transformed stream. Every transform corresponds to an action an attacker on the
host can perform; nothing deletes or edits recorded events of other principals.

  --adversary='{"rename": true}'
      Attacker-created artifacts (attack processes/files whose label never occurs among benign
      nodes) are given the most frequent benign label of their type: the attacker names its
      dropped binary / files after common benign ones.
  --adversary='{"borrow": true}'
      Events issued by attack processes are re-attributed to the busiest benign process of the
      day: the attacker injects code into a long-lived benign process and acts from it.
  --adversary='{"substitute": true}'
      Attack syscalls are replaced by semantically equivalent ones within the model's edge
      vocabulary (SENDTO<->SENDMSG, RECVFROM<->RECVMSG).
  --adversary='{"dilute": n}'
      For every attack event, the attack process also reads n frequent benign files just before
      it: benign noise interleaved with the attack.
  --adversary='{"prime": h}'
      Each attack process reads frequent benign files at a steady rate during the h hours before
      its attack starts: it builds a benign-looking history first (temporal-memory priming).
Keys can be combined. The spec enters the construction hash only when set (pipeline.py).
"""

import json
from collections import Counter

EQUIV = {"EVENT_SENDTO": "EVENT_SENDMSG", "EVENT_SENDMSG": "EVENT_SENDTO",
         "EVENT_RECVFROM": "EVENT_RECVMSG", "EVENT_RECVMSG": "EVENT_RECVFROM"}
NS_H = 3600 * 10**9


def spec_of(cfg):
    s = getattr(cfg, "_adversary", "") or ""
    return json.loads(s) if s else None


def prepare(cfg, indexid2msg):
    """Apply renaming to the label table and build the context for event transforms."""
    spec = spec_of(cfg)
    if not spec:
        return indexid2msg, None
    from pidsmaker.utils.labelling import get_ground_truth, ground_truth_attacks
    from pidsmaker.utils.utils import log

    gt, _, _ = get_ground_truth(cfg)
    gt = {int(n) for n in gt}
    benign_labels = Counter()
    for nid, (ntype, label) in indexid2msg.items():
        if int(nid) not in gt:
            benign_labels[(ntype, label)] += 1
    top_label = {}
    for (ntype, label), c in benign_labels.most_common():
        top_label.setdefault(ntype, label)
    seen_benign = {lab for (_, lab) in benign_labels}
    indexid2msg = dict(indexid2msg)
    renamed = 0
    if spec.get("rename"):
        for nid in gt:
            if nid in indexid2msg:
                ntype, label = indexid2msg[nid]
                if ntype in ("subject", "file") and label not in seen_benign and ntype in top_label:
                    indexid2msg[nid] = (ntype, top_label[ntype])
                    renamed += 1
    # frequent benign files (targets for injected reads), by label frequency
    files = [nid for nid, (t, l) in indexid2msg.items()
             if t == "file" and int(nid) not in gt and benign_labels[(t, l)] > 1]
    files.sort(key=lambda n: -benign_labels[tuple(indexid2msg[n])])
    attack_procs = sorted(n for n in gt if n in indexid2msg and indexid2msg[n][0] == "subject")
    attacks = [(int(s), int(e)) for _, s, e in ground_truth_attacks(cfg)]
    ctx = {"spec": spec, "gt": gt, "files": files[:200], "attack_procs": attack_procs,
           "attacks": attacks, "borrower": None, "n_syn": 0, "test_dates": set(cfg.dataset.test_dates)}
    ctx["prime_rows"] = _priming_rows(ctx, spec.get("prime", 0))
    log(f"[adversary] spec={spec} renamed={renamed} attack_procs={len(attack_procs)} "
        f"benign_file_pool={len(ctx['files'])} priming_events={len(ctx['prime_rows'])}")
    return indexid2msg, ctx


def _syn(ctx, src, dst, op, t):
    ctx["n_syn"] += 1
    k = ctx["n_syn"]
    return ("adv", src, op, "adv", dst, f"adv-{k}", int(t), -k)


def _priming_rows(ctx, hours):
    if not hours or not ctx["files"] or not ctx["attack_procs"]:
        return []
    rows = []
    for start, _ in ctx["attacks"]:
        n = int(hours * 60)                       # one read per minute per attack process
        for p in ctx["attack_procs"]:
            for i in range(n):
                t = start - hours * NS_H + i * (hours * NS_H // max(n, 1))
                rows.append(_syn(ctx, p, ctx["files"][i % len(ctx["files"])], "EVENT_READ", t))
    rows.sort(key=lambda r: r[6])
    return rows


def _in_attack(ctx, t):
    return any(s <= t <= e for s, e in ctx["attacks"])


def transform(cfg, rows, date):
    """Transform one time-ordered chunk of event rows of `date` (8-tuples as in event_table)."""
    ctx = getattr(cfg, "_adv_ctx", None)
    if ctx is None or not rows or date not in ctx["test_dates"]:
        return rows
    spec, gt = ctx["spec"], ctx["gt"]
    if spec.get("borrow") and ctx["borrower"] is None:
        cnt = Counter(r[1] for r in rows if r[1] not in gt)
        if cnt:
            ctx["borrower"] = (cnt.most_common(1)[0][0])
    out = []
    n_dil = int(spec.get("dilute", 0) or 0)
    for r in rows:
        src_node, src, op, dst_node, dst, ev, t, _id = r
        attack_edge = (src in gt or dst in gt) and _in_attack(ctx, t)
        if attack_edge and src in gt:
            if n_dil and ctx["files"]:
                for i in range(n_dil):
                    out.append(_syn(ctx, src, ctx["files"][(ctx["n_syn"] + i) % len(ctx["files"])],
                                    "EVENT_READ", t - (n_dil - i)))
            if spec.get("substitute") and op in EQUIV:
                op = EQUIV[op]
            if spec.get("borrow") and ctx["borrower"] is not None:
                src_node, src = "adv-borrow", ctx["borrower"]
        out.append((src_node, src, op, dst_node, dst, ev, t, _id))
    if ctx["prime_rows"]:
        lo, hi = rows[0][6], rows[-1][6]
        extra = [p for p in ctx["prime_rows"] if lo <= p[6] <= hi]
        if extra:
            out = sorted(out + extra, key=lambda r: r[6])
    return out
