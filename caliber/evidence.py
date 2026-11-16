"""Evidence CALIBER sends with a case (paper: "Evidence with every case").

For one unit (time window, entity) of the test stream: how many of its distinct interactions are
new to the detector (novelty above benign validation quantiles) against how many routine activity
would show, the most new interactions, and its rarest actions with their counts on the benign
validation days.

    python -m caliber.evidence --artifact_dir A --system kairos --dataset THEIA_E3 --attack 1

prints the evidence for the attack's case CALIBER ranks highest and for the benign cases the
detector alone ranks first that day.
"""
import argparse

import numpy as np
import pandas as pd

from caliber import abstain_score as asc
from caliber import data
from caliber.lineage import case_keys
from caliber.online import day_of
from caliber.tiers import _with_borrower, unit_attacks

LEVELS = (0.9, 0.99, 0.999, 0.9999)


class Evidence:
    """Edge-level novelty and action counts for one detector/dataset, reusable across units."""

    def __init__(self, runs, meta, seeds):
        rv, pv = asc._edge_stats(runs, "val", False)
        self.rt, pt = asc._edge_stats(runs, "test", False)
        nv = sum(pv[s]["nov"] for s in seeds) / sum(pv[s]["n"] for s in seeds)
        self.nt = sum(pt[s]["nov"] for s in seeds) / sum(pt[s]["n"] for s in seeds)
        self.ref = np.sort(nv)
        self.typ = pd.Series(meta["type"].astype(str).values, index=meta["node"].astype(np.int64).values)
        self.path = pd.Series(meta["path"].fillna("").astype(str).values,
                              index=meta["node"].astype(np.int64).values)
        self.spath = self.path[self.typ.reindex(self.path.index).values == "subject"]
        self.counts = pd.Series(self._actions(rv)).value_counts()

    def _actions(self, r):
        """Action = (source executable or source type, event type, target type), as for rarity."""
        s, d = r["srcnode"].astype(np.int64), r["dstnode"].astype(np.int64)
        sl = s.map(self.spath).fillna("")
        sl = np.where(sl.values != "", sl.values, "T:" + s.map(self.typ).fillna("?").values)
        return pd.Series(sl).astype(str).values + "|" + r["edge_type"].astype(str).values + "|" + \
            d.map(self.typ).fillna("?").values

    def name(self, n):
        return self.path.get(int(n), "") or self.typ.get(int(n), "?")

    def of_unit(self, w, node, top=8):
        """Evidence dict for unit (window w, entity node)."""
        rt = self.rt
        sel = (rt["b"].values == w) & ((rt["srcnode"].values == node) | (rt["dstnode"].values == node))
        d = rt[sel].copy()
        d["nov"] = self.nt[sel]
        d["partner"] = np.where(d["srcnode"].values == node, d["dstnode"].values, d["srcnode"].values)
        d["dir"] = np.where(d["srcnode"].values == node, "->", "<-")
        d["action"] = self._actions(d)
        inter = d.groupby(["partner", "edge_type", "dir"], as_index=False)["nov"].max()
        inter["share_below"] = np.searchsorted(self.ref, inter["nov"].values) / len(self.ref)
        m = len(inter)
        acts = pd.Series(d["action"].unique())
        cnt = acts.map(self.counts).fillna(0).astype(int)
        return {
            "entity": self.name(node), "interactions": m,
            "new": {L: int((inter["share_below"] > L).sum()) for L in LEVELS},
            "expected_if_routine": {L: (1 - L) * m for L in LEVELS},
            "most_new": [(r["dir"], str(r["edge_type"]), self.name(r["partner"]), float(r["share_below"]))
                         for _, r in inter.sort_values("nov", ascending=False).head(top).iterrows()],
            "rarest": sorted(zip(acts, cnt), key=lambda z: z[1])[:top],
            "benign_events": int(self.counts.sum()),
        }


def show(ev, label):
    print(f"-- {label}: {ev['entity']}")
    exp = ev["expected_if_routine"]
    print(f"   {ev['interactions']} distinct interactions; new beyond benign "
          + ", ".join(f"{int(L * 1e4) / 100:g}%: {ev['new'][L]} (routine {exp[L]:.2g})" for L in LEVELS))
    for d, et, p, sh in ev["most_new"]:
        print(f"     new  {d} {et:6s} {p[:60]:60s} above {100 * sh:.3f}% of benign")
    for a, c in ev["rarest"]:
        print(f"     rare {a[:80]:80s} {c} times in {ev['benign_events']} benign events")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--attack", type=int, required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--benign", type=int, default=3, help="detector's top benign cases to show")
    a = ap.parse_args()
    A = a.artifact_dir
    pl = asc.density_runs(A, a.system, a.dataset)["none"]
    seeds = tuple(s for s in map(int, a.seeds.split(",")) if s in pl and pl[s].get("test_epochs"))
    base = data.load_manifests(A, a.system, a.dataset, asc.BASE_GROUP)
    m0 = sorted(base, key=lambda m: m["seed"])[0] if base else pl[seeds[0]]
    labels, meta = data.labels_for(m0, A)
    labels = _with_borrower(labels, m0, A)
    st = asc.scored_stream("abstain", A, a.system, a.dataset, labels, seeds=seeds)
    S, b = st["S"], st["views"]["b"][1]
    keys, _ = case_keys(st, A, a.dataset, meta)
    h = (st["t1"] // asc.HOUR).astype(np.int64)
    _, inv = np.unique(np.stack([keys.astype(np.int64), h]), axis=1, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    c_open = np.full(n, np.inf)
    np.minimum.at(c_open, inv, st["t1"])
    c_day = day_of(c_open, labels.get("timezone", "US/Eastern"))

    def cmax(s):
        cs = np.full(n, -np.inf)
        np.maximum.at(cs, inv, np.nan_to_num(s, nan=-np.inf))
        return cs
    cb, cS = cmax(b), cmax(S)
    y, node = st["y"], st["node"]
    ua = unit_attacks(node, st["t1"], np.flatnonzero(y), labels["attacks"])
    units = [i for i, ks in ua.items() if a.attack in ks]
    rank = lambda cs, c: int((cs[np.flatnonzero(c_day == c_day[c])] > cs[c]).sum()) + 1
    c = min({inv[i] for i in units}, key=lambda c_: rank(cS, c_))
    ix = np.flatnonzero(c_day == c_day[c])
    ev = Evidence({s: pl[s] for s in seeds}, meta, seeds)
    i = max([i for i in units if inv[i] == c], key=lambda j: S[j])
    show(ev.of_unit(st["w"][i], int(node[i])), f"attack case, rank detector {rank(cb, c)} CALIBER {rank(cS, c)}")
    mal = {inv[j] for j in np.flatnonzero(y)}
    for cc in [x for x in ix[np.argsort(-cb[ix])] if x not in mal][:a.benign]:
        iu = np.flatnonzero(inv == cc)
        j = iu[np.argmax(np.nan_to_num(b[iu], nan=-np.inf))]
        show(ev.of_unit(st["w"][j], int(node[j])),
             f"benign case, rank detector {rank(cb, cc)} CALIBER {rank(cS, cc)}")


if __name__ == "__main__":
    main()
