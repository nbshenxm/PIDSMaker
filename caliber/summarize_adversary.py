"""Compact summary of caliber.adversary reports.

  python -m caliber.summarize_adversary --artifact_dir A [--budget 50]
"""

import argparse
import collections
import glob
import json
import os

PARAMS = ["rate_h", "entities", "hours", "lead_h", "level", "days", "max_new_entities_h",
          "offset_h", "minus_sd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--budget", type=int, default=50)
    a = ap.parse_args()
    rep = os.path.join(a.artifact_dir, "caliber", "reports")
    for f in sorted(glob.glob(os.path.join(rep, f"adversary_*_B{a.budget}.json"))):
        rows = json.load(open(f, encoding="utf-8"))
        print("=== " + os.path.basename(f))
        groups = collections.OrderedDict()
        for r in rows:
            key = (r["scenario"], tuple((k, r[k]) for k in PARAMS if k in r))
            groups.setdefault(key, {})[r["policy"]] = r
        timing = collections.defaultdict(list)
        for (scn, params), pol in groups.items():
            if scn == "S3-timing":
                for p, r in pol.items():
                    timing[p].append(r["attack_entities_hit"])
                continue
            st, ca, nd = pol["static"], pol["caliber"], pol["nodedupe"]
            rl = pol.get("caliber_rl", ca)
            ps = " ".join(f"{k}={v}" for k, v in params)
            rev = lambda r: f"{r.get('reviewed_fifo', -1):3d}"  # noqa: E731
            print(f"  {scn:11s} {ps:52s} hit S/C/N/RL={st['attack_entities_hit']:3d}/"
                  f"{ca['attack_entities_hit']:3d}/{nd['attack_entities_hit']:3d}/"
                  f"{rl['attack_entities_hit']:3d} of {st['attack_entities']:3d} | "
                  f"reviewed@B(fifo) S/C/RL={rev(st)}/{rev(ca)}/{rev(rl)} | attacker inc/day "
                  f"S={st['attacker_incidents_per_day']:7.1f} C={ca['attacker_incidents_per_day']:7.1f} "
                  f"RL={rl['attacker_incidents_per_day']:7.1f}")
        for p, v in timing.items():
            v = sorted(v)
            print(f"  S3-timing   {p:9s} attack entities hit over {len(v)} offsets: "
                  f"min={v[0]} median={v[len(v) // 2]} max={v[-1]}")


if __name__ == "__main__":
    main()
