"""Abstention-aware unit scorer (design frozen 2026-11-13, notes/2026-11-13_paper_direction.md;
revisions: 1 majority-of-runs unanimity, 2 lagged novelty history, 3 abstention members (ABMEM),
4 collapse guard (SPEC_WARM), 5 prevalence channel and four-channel sum (PREV)).

Inputs: per detector and dataset, two density-enabled training configurations with the same seeds:
  abstaining decoder   --abstain {"o":3} --density   (gambler's-loss abstention + novelty column)
  plain decoder        --density                     (novelty column)
and the plain decoder's default runs (group da39a3ee5e) for its surprise channel b (falls back to
the plain+density runs when the default group lacks a seed). Members = seeds x saved epochs >= SNAP.

Edge level (abstaining decoder, members of one seed = its snapshots):
  collapse guard         (rev. 4) a seed whose abstaining run abstains on > COLLAPSE of validation
                         events at every snapshot is replaced by its warm-up run (SPEC_WARM)
  abstention members     (rev. 3) a snapshot votes on abstention only once it has learned to abstain:
                         its validation abstention share >= ABMEM x the run's peak share over its
                         snapshots (label-free; a slowly learning run would otherwise veto with its
                         early snapshots). Surprise and novelty use all snapshots >= SNAP.
  consensus abstention   every member of a seed abstains (p > THR); for pools of several seeds,
                         a majority of seeds abstain unanimously (within-seed unanimity)
  q_any                  mean surprise where no member abstains, else 0
  q_hier (pools)         q_any within each seed, then the mean over seeds
Unit level (unit = entity x batch window, max over its edges), each channel mapped to the
validation tail -log P_val(X >= x) (exponential tail beyond the top 1%, no saturation):
  b       plain decoder surprise (robust-z snapshot mean, as ensemble_stream)
  q       q_any (one seed) / q_hier (several seeds)
  mlvmax  bulk novelty inside the consensus-abstained set: per unit, over its abstained edges
          deduplicated by (partner, edge type), k_L = # more novel than the level-L quantile of
          validation abstained edges of the same type (Mondrian; global quantile for types with
          < MIN_TYPE validation edges), sign test -log P(Binom(m, 1-L) >= k_L), max over LEVELS,
          tail-calibrated on validation units; computed for the abstaining and the plain decoder's
          novelty and combined by max (recalibrated), then taken relative to the entity's own history:
          minus the largest novelty it showed in windows ending >= LAG_H (24 h) earlier (lag_excess)
  rar     (rev. 5) prevalence of the unit's actions in the benign validation days (NoDoze-style):
          sum of -log smoothed frequency of its PREV_K rarest distinct (source executable, event
          type, target type) actions (see prevalence)
Ranking (rev. 5, PREV=1): F = b + q + mlvmax + rar, a Fisher-style sum of four validation -log p;
r = tail_F(F) is the scalar for the online controller. Per-day ratio to the plain detector over the
12 development pairs: 0.39 (rev. 4: 0.52); PREV_K 3/5/10/all -> 0.44/0.39/0.39/0.43, 1 -> 0.64.
(PREV=0, rev. 4: round-robin of S = b + q + mlvmax and C = b + q; r = max(tail_S(S), tail_C(C)).)
Run-level interval: the scorer applied to each seed alone gives r_m; U = r + KAPPA sd_m(r_m)/sqrt(R).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import binom

from caliber import data
from caliber.lineage import case_keys
from caliber.pilot import normaliser, unit_labels, unit_table

SNAP = int(os.environ.get("CALIBER_SNAP", "3"))
LAST = int(os.environ.get("CALIBER_LAST", "0"))   # > 0: only the last LAST saved snapshots of a run
THR = 0.5
LEVELS = (0.5, 0.75, 0.9, 0.95, 0.99)
MIN_TYPE = 50
SPEC_ABSTAIN = os.environ.get("CALIBER_ABSTAIN_SPEC", '{"o":3}')
BASE_GROUP = "da39a3ee5e"
KAPPA = 1.645
LAG_H = 24.0          # novelty history lag: one review day
HOUR = 3600 * 10**9


TAILCAP = int(os.environ.get("CALIBER_TAILCAP", "0"))
# (rev. 5) prevalence channel and Fisher-style combination: score = b + q + nov + rar,
# each a validation -log p; replaces the round-robin of S = b + q + nov and C = b + q.
PREV = int(os.environ.get("CALIBER_PREV", "1"))
PREV_K = int(os.environ.get("CALIBER_PREV_K", "5"))


def prevalence(rv, rt, meta, uv_idx, ut_idx, k=None):
    """Prevalence of a unit's actions in the benign validation days (NoDoze-style rarity): an action
    is (source executable, or source type when the executable is unknown; event type; target type);
    its rarity is -log of its smoothed validation frequency; a unit scores the sum of the rarities of
    its k rarest distinct actions. Returns validation-tail -log p for (validation, test) units."""
    k = PREV_K if k is None else k
    typ = pd.Series(meta["type"].astype(str).values, index=meta["node"].astype(np.int64).values)
    path = pd.Series(meta["path"].fillna("").astype(str).values, index=meta["node"].astype(np.int64).values)
    path = path[typ.reindex(path.index).values == "subject"]

    def keys(r):
        s, d = r["srcnode"].astype(np.int64), r["dstnode"].astype(np.int64)
        sl = s.map(path).fillna("")
        sl = np.where(sl.values != "", sl.values, "T:" + s.map(typ).fillna("?").values)
        return pd.Series(sl).astype(str) + "|" + r["edge_type"].astype(str).values + "|" + d.map(typ).fillna("?").values

    kv, kt = keys(rv), keys(rt)
    cnt = kv.value_counts()
    n, V = len(kv), len(cnt) + 1

    def per_unit(r, kk):
        rr = -np.log((kk.map(cnt).fillna(0).values + 1.0) / (n + V))
        parts = [pd.DataFrame({"w": r["b"].values, "node": r[c].values, "k": kk.values, "r": rr})
                 for c in ("srcnode", "dstnode")]
        d = pd.concat(parts, ignore_index=True).drop_duplicates(["w", "node", "k"])
        d = d.sort_values("r", ascending=False).groupby(["w", "node"]).head(k)
        return d.groupby(["w", "node"])["r"].sum()

    sv = per_unit(rv, kv)
    f = tail_fn(sv.values)
    return f(sv.reindex(uv_idx).fillna(0).values), f(per_unit(rt, kt).reindex(ut_idx).fillna(0).values)


def tail_fn(ref, cap=False):
    """-log P_ref(X >= x), with an exponential tail fitted to the top 1% beyond it. cap (with
    TAILCAP): a channel claims at most the evidence its validation set resolves, -log 1/(n+1);
    units beyond it are then ordered by the other channels of the sum."""
    r = np.sort(ref[np.isfinite(ref)])
    n = len(r)
    if n < 50:
        return lambda x: np.zeros(len(x))
    if cap and TAILCAP:
        f0, top = tail_fn(ref), float(np.log(n + 1.0))
        return lambda x: np.minimum(f0(x), top)
    hi = r[int(0.99 * n):]
    lam = 1.0 / max(np.mean(hi - hi[0]), 1e-9)
    # empirical tail (n - #{ref < x} + 1)/(n + 1) at the validation values, linearly interpolated
    # between them (rev. 5b): a step function would tie every test score that falls between two
    # consecutive validation scores (tens of test units per validation unit), and ties are broken
    # at random downstream. Beyond the largest validation score: 1/(n+1) decaying at the fitted rate.
    u, first = np.unique(r, return_index=True)
    pu = (n - first + 1.0) / (n + 1.0)

    def f(x):                                        # in log space: no underflow, no top ties
        x = np.asarray(x, np.float64)
        lp = np.log(np.interp(x, u, pu))
        lp = np.where(x > u[-1], -lam * (x - u[-1]) - np.log(n + 1.0), lp)
        lext = np.log(0.01) - lam * np.maximum(x - hi[0], 0)
        lp = np.where(x > hi[0], np.minimum(lp, lext), lp)
        return -lp
    return f


# ----------------------------------------------------------------------------- inputs
def density_runs(artifact_dir, system, dataset):
    """{abstain spec ('none' = plain): {seed: manifest}} for density-enabled runs with test losses."""
    out = {}
    for m in data.load_manifests(artifact_dir, system, dataset):
        argv = m.get("argv", [])
        if "--density" not in argv or not m.get("test_epochs") or "--adversary" in argv:
            continue
        sp = argv[argv.index("--abstain") + 1].replace(" ", "") if "--abstain" in argv else "none"
        out.setdefault(sp, {}).setdefault(m["seed"], m)
    return out


def snapshots(m, snap=None):
    snap = SNAP if snap is None else snap
    eps = [e for e in sorted(set(m["val_epochs"]) & set(m["test_epochs"])) if e >= snap]
    return eps[-LAST:] if LAST else eps


ABMEM = float(os.environ.get("CALIBER_ABMEM", "0.5"))  # > 0: see abstain_snapshots
_ABMEM_CACHE = {}


_SHARE_CACHE = {}


def _val_share(m, e):
    """Share of validation edges snapshot e of an abstaining run abstains on (p > THR)."""
    key = (m["edge_losses_dir"], e)
    if key not in _SHARE_CACHE:
        _SHARE_CACHE[key] = float((data.read_split(m["edge_losses_dir"], "val", e, None)["abstain"].values
                                   > THR).mean())
    return _SHARE_CACHE[key]


# Collapse guard. Before a model predicts better than 1/o, abstaining on every event is the
# gambler's-loss optimum (loss log o); a slowly learning detector can lock into it from the first
# steps and then abstains on everything (Kairos on ATLAS). Such a run carries no abstention signal.
# A run whose every snapshot abstains on more than COLLAPSE of the validation events is replaced by
# the same seed trained with a cross-entropy warm-up (SPEC_WARM: plain cross-entropy until the running
# training loss falls below log o). Label-free; runs that do not collapse are untouched.
SPEC_WARM = '{"o":3,"warm":"auto"}'
COLLAPSE = 0.99


def collapsed(m):
    eps = snapshots(m)
    return bool(eps) and all(_val_share(m, e) > COLLAPSE for e in eps)


def abstain_runs(runs, seeds, spec=None):
    """{seed: manifest} of the abstaining runs to score, with the collapse guard applied when the
    default configuration is used, and the seeds that fell back to SPEC_WARM."""
    spec = SPEC_ABSTAIN if spec is None else spec
    ab = runs.get(spec, {})
    if spec != '{"o":3}' or os.environ.get("CALIBER_GUARD", "1") == "0":
        return {s: ab[s] for s in seeds if s in ab}, ()
    warm = runs.get(SPEC_WARM, {})
    out, fell = {}, []
    for s in seeds:
        m = ab.get(s)
        if (m is None or collapsed(m)) and s in warm:
            if m is not None:
                print(f"  collapse guard: seed {s} abstains on > {COLLAPSE} of validation events at every "
                      f"snapshot -> warm-up run", flush=True)
            out[s] = warm[s]
            fell.append(s)
        elif m is not None:
            out[s] = m
    return out, tuple(fell)


def abstain_snapshots(m):
    """Snapshots of an abstaining run that vote on abstention. With ABMEM = f > 0, a snapshot
    votes only once it has learned to abstain: its validation abstention share is at least f
    times the run's peak share over its snapshots (label-free; drops snapshots taken before the
    abstaining objective engaged or converged; keeps all snapshots when the share is stable)."""
    eps = snapshots(m)
    if ABMEM <= 0 or len(eps) < 2:
        return eps
    key = (m["edge_losses_dir"], tuple(eps))
    if key not in _ABMEM_CACHE:
        share = {e: _val_share(m, e) for e in eps}
        peak = max(share.values())
        keep = [e for e in eps if peak > 0 and share[e] >= ABMEM * peak] or eps
        print(f"  abstain members seed {m.get('seed')}: shares "
              + " ".join(f"{e}:{share[e]:.3f}" for e in eps) + f" -> {keep}", flush=True)
        _ABMEM_CACHE[key] = keep
    return _ABMEM_CACHE[key]


def _variant():
    """Cache-key suffix for non-default members or abstaining configuration."""
    if (SNAP, LAST, SPEC_ABSTAIN, ABMEM, TAILCAP, PREV, PREV_K) == (3, 0, '{"o":3}', 0.5, 0, 1, 5):
        return ""
    import hashlib
    key = f"{SNAP}|{LAST}|{SPEC_ABSTAIN}|{ABMEM}" + (f"|cap{TAILCAP}" if TAILCAP else "") \
        + (f"|prev{PREV}k{PREV_K}")
    return "_" + hashlib.md5(key.encode()).hexdigest()[:8]


def _edge_stats(ms, split, with_abstain):
    """Per seed, streamed over its snapshots: consensus abstention, any-abstention, surprise sum,
    novelty sum, member count. Returns (ref edges, {seed: stats})."""
    ref, per = None, {}
    for s, m in ms.items():
        st = None
        for e in (abstain_snapshots(m) if with_abstain else snapshots(m)):
            df = data.read_split(m["edge_losses_dir"], split, e, None)
            if ref is None:
                ref = df[["srcnode", "dstnode", "time", "b", "edge_type"]].reset_index(drop=True)
            else:
                assert len(df) == len(ref) and (df["time"].values == ref["time"].values).all() \
                    and (df["srcnode"].values == ref["srcnode"].values).all(), "edge streams differ"
            q = df["loss"].values.astype(np.float32)
            nov = df["novelty"].values.astype(np.float32)
            ab = (df["abstain"].values > THR) if with_abstain else np.zeros(len(df), bool)
            if st is None:
                st = {"all": ab.copy(), "any": ab.copy(), "q": q.astype(np.float64),
                      "nov": nov.astype(np.float64), "n": 1,
                      "qk": np.where(ab, 0.0, q).astype(np.float64), "nk": (~ab).astype(np.int16)}
            else:
                st["all"] &= ab
                st["any"] |= ab
                st["q"] += q
                st["nov"] += nov
                st["n"] += 1
                st["qk"] += np.where(ab, 0.0, q)
                st["nk"] += ~ab
            del df
        if st is not None:
            per[s] = st
    return ref, per


def units(ref, cols):
    """Unit table (w = batch, node) with max per channel and the partner of the max-q edge."""
    parts = []
    for nc, pc in (("srcnode", "dstnode"), ("dstnode", "srcnode")):
        d = {"node": ref[nc].values, "partner": ref[pc].values, "w": ref["b"].values,
             "time": ref["time"].values}
        d.update(cols)
        parts.append(pd.DataFrame(d))
    long = pd.concat(parts, ignore_index=True)
    g = long.groupby(["w", "node"])
    u = pd.DataFrame({"t0": g["time"].min(), "t1": g["time"].max()})
    for c in cols:
        u[c] = g[c].max()
    first = list(cols)[0]
    u["partner"] = long.loc[g[first].idxmax().values, "partner"].values
    return u


# ----------------------------------------------------------------------------- bulk novelty
def _sign_counts(ref, gate, nov, thr):
    parts = []
    for nc, pc in (("srcnode", "dstnode"), ("dstnode", "srcnode")):
        parts.append(pd.DataFrame({"w": ref["b"].values[gate], "node": ref[nc].values[gate],
                                   "partner": ref[pc].values[gate],
                                   "et": ref["edge_type"].values[gate], "nov": nov[gate]}))
    d = pd.concat(parts, ignore_index=True)
    d = d.groupby(["w", "node", "partner", "et"], as_index=False)["nov"].max()
    out = {"m": d.groupby(["w", "node"]).size()}
    for L in LEVELS:
        t = d["et"].map(thr[L]).fillna(np.inf).values
        out[L] = pd.Series((d["nov"].values > t).astype(np.int32)).groupby(
            [d["w"].values, d["node"].values]).sum()
    o = pd.DataFrame(out)
    o.index.names = ["w", "node"]
    return o


def _mlv(df):
    m = df["m"].values
    return np.max([np.where(m > 0, -binom.logsf(df[L].values - 1, m, 1 - L), 0.0) for L in LEVELS], 0)


def bulk_novelty(rv, gv, nv, rt, gt, nt, uv_idx, ut_idx):
    """Multilevel sign test over consensus-abstained edges; (validation, test) unit tails."""
    dv = pd.DataFrame({"et": rv["edge_type"].values[gv], "nov": nv[gv]})
    thr = {}
    for L in LEVELS:
        g = dv.groupby("et")["nov"].quantile(L)
        cnt = dv.groupby("et").size()
        glob = np.quantile(nv[gv], L) if gv.any() else np.inf
        thr[L] = {t: (g[t] if cnt[t] >= MIN_TYPE else glob) for t in g.index}
    sv = _sign_counts(rv, gv, nv, thr).reindex(uv_idx).fillna(0)
    st = _sign_counts(rt, gt, nt, thr).reindex(ut_idx).fillna(0)
    mv, mt = _mlv(sv), _mlv(st)
    f = tail_fn(mv)
    return f(mv), f(mt)


def lag_excess(node_v, t_v, nov_v, node_t, t_t, nov_t, lag_h=None):
    """Novelty relative to the entity's own history: nov minus the largest novelty the same entity
    showed in windows ending at least lag_h hours earlier (validation, then test), floored at 0.
    Novelty surfaced a review day ago was delivered then; repeating it every day is routine (a
    persistent novel benign entity otherwise tops every day). Recent repeats (an intrusion's
    consecutive windows) keep their credit."""
    lag_h = LAG_H if lag_h is None else lag_h
    nodes = np.r_[node_v, node_t]
    t = np.r_[t_v, t_t].astype(np.float64)
    nv = np.r_[nov_v, nov_t].astype(np.float64)
    rank = pd.factorize(nodes)[0].astype(np.float64)
    span = t.max() - t.min() + 10 * 86400e9
    key = rank * span + (t - t.min())
    o = np.argsort(key, kind="stable")
    ks, vs, gs = key[o], nv[o], rank[o]
    cm = pd.Series(vs).groupby(gs).cummax().values
    pos = np.searchsorted(ks, ks - lag_h * 3600e9, side="right") - 1
    ok = (pos >= 0) & (gs[np.maximum(pos, 0)] == gs)
    prev = np.where(ok, cm[np.maximum(pos, 0)], 0.0)
    ex = np.empty(len(nv))
    ex[o] = np.maximum(vs - np.maximum(prev, 0.0), 0.0)
    return ex[:len(node_v)], ex[len(node_v):]



# ----------------------------------------------------------------------------- plain surprise
def plain_unit_scores(ms):
    """Plain decoder: robust-z (own validation units) unit max per snapshot member, mean per seed.
    Returns ({seed: Series val}, {seed: Series test}) indexed by (w, node)."""
    val, test = {}, {}
    for s, m in ms.items():
        zv, zt = [], []
        for e in snapshots(m):
            v = unit_table(data.read_split(m["edge_losses_dir"], "val", e, None), "entity_window")
            t = unit_table(data.read_split(m["edge_losses_dir"], "test", e, None), "entity_window")
            f = normaliser(v["score"].values, "robust_z")
            zv.append(pd.Series(f(v["score"].values), index=v.index))
            zt.append(pd.Series(f(t["score"].values), index=t.index))
        val[s] = pd.concat(zv, axis=1).mean(1)
        test[s] = pd.concat(zt, axis=1).mean(1)
    return val, test


# ----------------------------------------------------------------------------- scorer
def _score(seeds, A, P, plain_v, plain_t, rv, rt, uv, ut, rar=None):
    """Frozen scorer on the seed pool `seeds`. Returns dict of (val, test) unit arrays. rar: the
    prevalence channel (validation, test) when PREV is on."""
    def pool(stats):
        ss = [stats[s] for s in seeds]
        # gate of the bulk-novelty test: one seed = unanimity over its snapshots; pooled = a majority
        # of runs abstain unanimously (flat unanimity over all members shrinks the set as runs are
        # added: pooled ATLAS rr 68.4 -> 39.9, others unchanged)
        all_ = np.mean([x["all"] for x in ss], 0) >= 0.5
        n = sum(x["n"] for x in ss)
        nov = sum(x["nov"] for x in ss) / n
        if len(ss) == 1:
            x = ss[0]
            q = np.where(x["any"], 0.0, x["q"] / x["n"])
        else:                                        # q_hier
            q = np.mean([np.where(x["any"], 0.0, x["q"] / x["n"]) for x in ss], 0)
        return all_, q, nov
    gv, qv, nov_v = pool(A["val"])
    gt, qt, nov_t = pool(A["test"])
    nplv = sum(P["val"][s]["nov"] for s in seeds) / sum(P["val"][s]["n"] for s in seeds)
    nplt = sum(P["test"][s]["nov"] for s in seeds) / sum(P["test"][s]["n"] for s in seeds)
    # q channel: unit max, validation tail
    quv = units(rv, {"q": qv})["q"].reindex(uv.index).fillna(0).values
    qut = units(rt, {"q": qt})["q"].reindex(ut.index).fillna(0).values
    fq = tail_fn(quv, cap=True)
    q_v, q_t = fq(quv), fq(qut)
    # bulk novelty, two views, max recalibrated
    vo, to = bulk_novelty(rv, gv, nov_v, rt, gt, nov_t, uv.index, ut.index)
    vp, tp = bulk_novelty(rv, gv, nplv, rt, gt, nplt, uv.index, ut.index)
    fm = tail_fn(np.maximum(vo, vp), cap=True)
    n_v, n_t = fm(np.maximum(vo, vp)), fm(np.maximum(to, tp))
    # novelty an entity already showed a review day earlier (>= LAG_H) was surfaced then: routine now
    n_v, n_t = lag_excess(uv.index.get_level_values("node").values, uv["t1"].values, n_v,
                          ut.index.get_level_values("node").values, ut["t1"].values, n_t)
    # plain surprise b: mean over seeds of the seed means (robust z), validation tail over the
    # plain decoder's own validation units; units it lacks score 0
    bv_all = pd.concat([plain_v[s] for s in seeds], axis=1).mean(1)
    bt_all = pd.concat([plain_t[s] for s in seeds], axis=1).mean(1)
    fb = tail_fn(bv_all.values, cap=True)
    b_v = fb(np.nan_to_num(bv_all.reindex(uv.index).values, nan=-1e9))
    b_t = fb(np.nan_to_num(bt_all.reindex(ut.index).values, nan=-1e9))
    S_v, S_t = b_v + q_v + n_v, b_t + q_t + n_t
    C_v, C_t = b_v + q_v, b_t + q_t
    fS, fC = tail_fn(S_v), tail_fn(C_v)
    if rar is not None:
        # rev. 5: Fisher-style sum of the four validation -log p; r is its validation tail
        F_v, F_t = S_v + rar[0], S_t + rar[1]
        fF = tail_fn(F_v)
        r_v, r_t = fF(F_v), fF(F_t)
    else:
        r_v = np.maximum(fS(S_v), fC(C_v))
        r_t = np.maximum(fS(S_t), fC(C_t))
    out = {"b": (b_v, b_t), "q": (q_v, q_t), "nov": (n_v, n_t), "S3": (S_v, S_t),
           "C": (C_v, C_t), "r": (r_v, r_t), "cons_share_test": float(gt.mean())}
    if rar is not None:
        out["rar"] = rar
        out["F"] = (F_v, F_t)
    return out


def stream(artifact_dir, system, dataset, labels, seeds=(0, 1, 2), base_group=BASE_GROUP,
           spec=SPEC_ABSTAIN, verbose=True):
    """Unit stream with the fields of online.ensemble_stream, scored by the frozen abstention-aware
    scorer. S = pooled scorer r, Z = per-seed r_m, U = S + KAPPA * sd(Z)/sqrt(R). Also returns the
    round-robin views (S3, C) and the channels for diagnostics under st['views']."""
    runs = density_runs(artifact_dir, system, dataset)
    ab, _ = abstain_runs(runs, seeds, spec)
    pl = {s: m for s, m in runs.get("none", {}).items() if s in seeds}
    seeds = tuple(s for s in seeds if s in ab and s in pl)
    if not seeds:
        raise FileNotFoundError(f"{system}/{dataset}: need {spec}+density and plain+density runs "
                                f"(have {sorted((k, sorted(v)) for k, v in runs.items())})")
    ab = {s: ab[s] for s in seeds}
    pl = {s: pl[s] for s in seeds}
    base = {m["seed"]: m for m in data.load_manifests(artifact_dir, system, dataset, base_group)
            if m["seed"] in seeds and m.get("test_epochs")}
    plain_src = {s: base.get(s, pl[s]) for s in seeds}
    A, P = {}, {}
    for split in ("val", "test"):
        r_a, A[split] = _edge_stats(ab, split, True)
        r_p, P[split] = _edge_stats(pl, split, False)
        assert len(r_a) == len(r_p) and (r_a["time"].values == r_p["time"].values).all()
        if split == "val":
            rv = r_a
        else:
            rt = r_a
        del r_p
    if verbose:
        print(f"  abstain-score {system}/{dataset}: seeds {seeds}, edges val {len(rv)} test {len(rt)}",
              flush=True)
    # units from the abstaining decoder's edges; the partner (lineage key) is the other endpoint
    # of the unit's most surprising edge, surprise = mean over the members that do not abstain
    def q_nonab(stats):
        qk = sum(stats[s]["qk"] for s in seeds)
        nk = sum(stats[s]["nk"].astype(np.int32) for s in seeds)
        return np.where(nk > 0, qk / np.maximum(nk, 1), 0.0)
    uv, ut = units(rv, {"q": q_nonab(A["val"])}), units(rt, {"q": q_nonab(A["test"])})
    plain_v, plain_t = plain_unit_scores(plain_src)
    rar = None
    if PREV:
        _, meta = data.labels_for(ab[seeds[0]], artifact_dir)
        rar = prevalence(rv, rt, meta, uv.index, ut.index)
    pooled = _score(seeds, A, P, plain_v, plain_t, rv, rt, uv, ut, rar)
    per = [_score((s,), A, P, plain_v, plain_t, rv, rt, uv, ut, rar) for s in seeds] if len(seeds) > 1 \
        else [pooled]
    Zv = np.vstack([p["r"][0] for p in per])
    Zt = np.vstack([p["r"][1] for p in per])
    R = len(per)
    sem_v = Zv.std(0, ddof=0) / np.sqrt(R) if R > 1 else np.zeros(Zv.shape[1])
    sem_t = Zt.std(0, ddof=0) / np.sqrt(R) if R > 1 else np.zeros(Zt.shape[1])
    y = unit_labels(ut, "entity_window", labels)
    if verbose:
        print(f"  consensus-abstained test edges {pooled['cons_share_test']:.3f}; units val {len(uv)} "
              f"test {len(ut)}; attack units {int(y.sum())}", flush=True)
    return {
        "S": pooled["r"][1], "S_val": pooled["r"][0],
        "U": pooled["r"][1] + KAPPA * sem_t, "U_val": pooled["r"][0] + KAPPA * sem_v,
        "Umax": np.maximum(pooled["r"][1], Zt.max(0)), "Umax_val": np.maximum(pooled["r"][0], Zv.max(0)),
        "Z": Zt, "Z_val": Zv,
        "t0": ut["t0"].values, "t1": ut["t1"].values,
        "w": ut.index.get_level_values("w").values, "y": y,
        "node": ut.index.get_level_values("node").values, "partner": ut["partner"].values,
        "t_val": uv["t1"].values, "node_val": uv.index.get_level_values("node").values,
        "partner_val": uv["partner"].values,
        "seeds": seeds,
        "views": {k: pooled[k] for k in ("b", "q", "nov", "S3", "C", "r", "rar", "F") if k in pooled},
        "views_per_seed": [{k: p[k] for k in ("b", "S3", "C", "r")} for p in per],
    }


# ----------------------------------------------------------------------------- offline check
def bstar(S, st, artifact_dir, dataset, meta, attacks, tz, keys=None):
    """Per-attack first-detection budget (cases/day) of a ranking of cases (lineage x hour, case
    score = max unit score). S: unit scores, or a list of unit scores for a round-robin merge of
    their case rankings. Returns ({attack: B*}, geometric mean with floor 1)."""
    from caliber.online import day_of
    from caliber.tiers import unit_attacks
    t1, y = st["t1"], st["y"]
    if keys is None:
        keys, _ = case_keys(st, artifact_dir, dataset, meta)
    hour = (t1 // HOUR).astype(np.int64)
    _, inv = np.unique(np.stack([keys.astype(np.int64), hour]), axis=1, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1

    def case_rank(Sc, seed):
        cs = np.full(n, -np.inf)
        np.maximum.at(cs, inv, np.nan_to_num(Sc, nan=-np.inf))
        o = np.lexsort((np.random.default_rng(seed).random(n), -cs))
        r = np.empty(n, np.float64)
        r[o] = np.arange(n)
        r[~np.isfinite(cs)] = np.inf
        return r
    if isinstance(S, (list, tuple)):
        Rk = np.vstack([case_rank(Sc, i) for i, Sc in enumerate(S)])
        cs = -(Rk.min(0) * len(S) + Rk.argmin(0))
        o = np.lexsort((np.random.default_rng(0).random(n), -cs))
        rank = np.empty(n, np.int64)
        rank[o] = np.arange(n)
    else:
        rank = case_rank(S, 0).astype(np.int64)
    n_days = len(np.unique(day_of(t1, tz)))
    ua = unit_attacks(st["node"], t1, np.flatnonzero(y), attacks)
    best = {}
    for i in np.flatnonzero(y):
        for a in ua.get(i, ()):
            best[a] = min(best.get(a, np.inf), (rank[inv[i]] + 1) / n_days)
    gm = float(np.exp(np.mean(np.log(np.maximum(list(best.values()), 1.0))))) if best else np.nan
    return best, gm


def as_plain(st):
    """The same units scored by the plain decoder's surprise channel b alone (matched baseline:
    same seeds, same snapshot members, same unit table)."""
    out = dict(st)
    bv, bt = st["views"]["b"]
    Zv = np.vstack([p["b"][0] for p in st["views_per_seed"]])
    Zt = np.vstack([p["b"][1] for p in st["views_per_seed"]])
    R = Zt.shape[0]
    sem_v = Zv.std(0, ddof=0) / np.sqrt(R) if R > 1 else 0.0
    sem_t = Zt.std(0, ddof=0) / np.sqrt(R) if R > 1 else 0.0
    out.update({"S": bt, "S_val": bv, "Z": Zt, "Z_val": Zv,
                "U": bt + KAPPA * sem_t, "U_val": bv + KAPPA * sem_v})
    return out


def scored_stream(kind, artifact_dir, system, dataset, labels, seeds=(0, 1, 2)):
    """kind: "abstain"/"abstain_dual" (frozen scorer) or "plain3" (b alone on the same units). The scored stream
    is cached under <artifact_dir>/caliber/cache (labels are re-applied on load)."""
    import pickle
    cdir = os.path.join(artifact_dir, "caliber", "cache")
    _, fell = abstain_runs(density_runs(artifact_dir, system, dataset), seeds)
    wtag = f"_w{'-'.join(map(str, fell))}" if fell else ""
    cp = os.path.join(cdir, f"abstain_score_v6_{system}_{dataset}_s{'-'.join(map(str, seeds))}"
                            f"{_variant()}{wtag}.pkl")
    st = None
    if os.path.exists(cp):
        with open(cp, "rb") as fh:
            st = pickle.load(fh)
        idx = pd.DataFrame({"t0": st["t0"], "t1": st["t1"]},
                           index=pd.MultiIndex.from_arrays([st["w"], st["node"]], names=["w", "node"]))
        st["y"] = unit_labels(idx, "entity_window", labels)
    if st is None:
        st = stream(artifact_dir, system, dataset, labels, seeds=seeds)
        os.makedirs(cdir, exist_ok=True)
        with open(cp + ".tmp", "wb") as fh:
            pickle.dump(st, fh, protocol=4)
        os.replace(cp + ".tmp", cp)
    R = st["Z"].shape[0]                            # interval over runs (recomputed on load)
    for sfx in ("", "_val"):
        sem = st["Z" + sfx].std(0) / np.sqrt(R) if R > 1 else 0.0
        st["U" + sfx] = st["S" + sfx] + KAPPA * sem
    return as_plain(st) if kind == "plain3" else st


def dual_views(st):
    """The two round-robin views (S = b + q + mlvmax, C = b + q) for the online form with one
    controller per view: per view (S_test, S_val, U_test, U_val), U = view + KAPPA sd_seeds/sqrt(R)."""
    out = []
    for name in ("S3", "C"):
        sv, stt = st["views"][name]
        Zv = np.vstack([p[name][0] for p in st["views_per_seed"]])
        Zt = np.vstack([p[name][1] for p in st["views_per_seed"]])
        R = Zt.shape[0]
        sem_v = Zv.std(0, ddof=0) / np.sqrt(R) if R > 1 else 0.0
        sem_t = Zt.std(0, ddof=0) / np.sqrt(R) if R > 1 else 0.0
        out.append((stt, sv, stt + KAPPA * sem_t, sv + KAPPA * sem_v))
    return out
