"""Go/no-go pilot: does seed-ensemble disagreement carry signal beyond the score?

Node-level analysis (matches PIDSMaker's node_evaluation: node score = max loss
over incident edges, node is malicious iff it is in the ground truth).

No test labels are used for any choice:
  * each member's epoch is selected by mean benign validation loss (or `last`);
  * validation nodes are split by node id into a NORMALISATION half (A), used to
    turn each member's raw score into a scale-free -log10 p-value, and a
    CALIBRATION half (B), used for ensemble conformal p-values / thresholds.

Signals per test node x, over members m = 1..M:
  z_m(x) = member-normalised score, fitted on member m's A-half scores:
           robust_z (default): (x - median_A) / (1.4826 * MAD_A)
           logp:               -log10 p_m(x)  (saturates above max(A): ties at the top)
  S(x)   = mean_m z_m(x)       suspicion   (ensemble score)
  U(x)   = std_m  z_m(x)       disagreement (normalised, so not a raw-scale artefact)
  V(x)   = fraction of members whose raw score exceeds their own max-val threshold

Usage (on the cluster, pids3 env):
  python -m caliber.pilot --artifact_dir /fs1/.../artifacts_v3 [--system velox] [--dataset CADETS_E3]
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from caliber import conformal, data

ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def _auc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


def _ap(y, s):
    y = np.asarray(y)
    if y.sum() == 0:
        return None
    return float(average_precision_score(y, s))


def _confusion(y, flag):
    y = np.asarray(y, bool)
    flag = np.asarray(flag, bool)
    tp = int((y & flag).sum())
    fp = int((~y & flag).sum())
    fn = int((y & ~flag).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec}


def _half(nodes):
    """Deterministic 50/50 split of node ids (hash, not parity, to avoid id-order bias)."""
    h = (np.asarray(nodes, dtype=np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
    return (h >> np.uint64(31)).astype(bool)


def cv_auc(X, y, seed=0, folds=5):
    """Cross-validated AUROC / AP of a logistic model on features X."""
    y = np.asarray(y)
    if y.sum() < folds:
        return None
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.decision_function(sc.transform(X[te]))
    return {"auroc": _auc(y, pred), "ap": _ap(y, pred)}


def normaliser(ref, kind="robust_z"):
    ref = np.asarray(ref, dtype=np.float64)
    if kind == "robust_z":
        med = np.median(ref)
        mad = 1.4826 * np.median(np.abs(ref - med))
        if mad <= 0:
            mad = ref.std() or 1.0
        return lambda x: (np.asarray(x, dtype=np.float64) - med) / mad
    if kind == "logp":
        return lambda x: -np.log10(conformal.pvalues(ref, x))
    raise ValueError(kind)


def unit_table(df, unit="node", window="batch", topk=5):
    """Scores per decision unit, with the unit's time span [t0, t1].

    node:          one row per entity over the whole split (PIDSMaker node evaluation)
    entity_window: one row per (window, entity)
    window:        one row per window; score = mean of its top-k entity scores
    """
    if unit == "node":
        t = data.node_scores(df)
        return t
    long = data._node_long(df)
    long["w"] = data.window_ids(long, window)
    g = long.groupby(["w", "node"])
    ew = pd.DataFrame({"score": g["loss"].max(), "t0": g["time"].min(), "t1": g["time"].max()})
    if unit == "entity_window":
        return ew
    if unit == "window":
        gw = ew.groupby(level="w")
        return pd.DataFrame({
            "score": gw["score"].apply(lambda s: np.sort(s.values)[::-1][:topk].mean()),
            "t0": gw["t0"].min(), "t1": gw["t1"].max(),
        })
    raise ValueError(unit)


def unit_labels(tbl, unit, labels, ew_tbl=None):
    """Ground truth per unit. Time-based units are malicious only when the entity is in
    an attack's ground truth AND the unit's time span overlaps that attack's interval."""
    if unit == "node":
        return np.isin(tbl.index.values, labels["malicious_nids"])
    if unit == "entity_window":
        nodes = tbl.index.get_level_values("node").values
        t0, t1 = tbl["t0"].values, tbl["t1"].values
        y = np.zeros(len(tbl), bool)
        for a in labels["attacks"]:
            y |= np.isin(nodes, a["nids"]) & (t1 >= a["start_ns"]) & (t0 <= a["end_ns"])
        return y
    if unit == "window":
        y_ew = unit_labels(ew_tbl, "entity_window", labels)
        mal_w = set(ew_tbl.index.get_level_values("w").values[y_ew])
        return np.isin(tbl.index.values, list(mal_w))
    raise ValueError(unit)


def _half_keys(index):
    if isinstance(index, pd.MultiIndex):
        w = index.get_level_values(0).values.astype(np.int64)
        n = index.get_level_values(1).values.astype(np.int64)
        return _half(np.abs(n * 1000003 + w))
    return _half(np.abs(index.values.astype(np.int64)))


def analyse_group(members, labels, epoch_rule="min_val_loss", cache_dir=None, log=print,
                  norm="robust_z", unit="node", window="batch"):
    per_member, z_test, z_cal, raw_test, vote = [], [], [], [], []
    test_index = None
    y = None

    for m in members:
        ep = data.select_epoch(m, epoch_rule, cache_dir)
        val = unit_table(data.read_split(m["edge_losses_dir"], "val", ep, cache_dir), unit, window)
        test_df = data.read_split(m["edge_losses_dir"], "test", ep, cache_dir)
        test = unit_table(test_df, unit, window)
        if test_index is None:
            test_index = test.index
            ew = unit_table(test_df, "entity_window", window) if unit == "window" else None
            y = unit_labels(test, unit, labels, ew)
        test = test.reindex(test_index)

        in_a = _half_keys(val.index)
        val_a, val_b = val["score"].values[in_a], val["score"].values[~in_a]

        thr_maxval = val["score"].max()            # PIDSMaker max_val_loss
        flag = test["score"].values > thr_maxval
        per_member.append({
            "seed": m["seed"], "epoch": int(ep),
            "auroc": _auc(y, test["score"].values), "ap": _ap(y, test["score"].values),
            "maxval": _confusion(y, flag), "threshold": float(thr_maxval),
        })
        log(f"  seed {m['seed']:>3} epoch {ep:>2}: AUROC={per_member[-1]['auroc']} "
            f"AP={per_member[-1]['ap']} maxval={per_member[-1]['maxval']}")

        f_norm = normaliser(val_a, norm)
        z_test.append(f_norm(test["score"].values))
        z_cal.append(f_norm(val_b))
        raw_test.append(test["score"].values)
        vote.append(flag)

    Zt, Zc = np.vstack(z_test), np.vstack(z_cal)
    S, U = Zt.mean(0), Zt.std(0)
    S_cal = Zc.mean(0)
    V = np.vstack(vote).mean(0)
    p_ens = conformal.pvalues(S_cal, S)
    n_cal = len(S_cal)

    out = {"n_members": len(members), "unit": unit, "window": window,
           "n_test_nodes": int(len(y)), "n_malicious_test_nodes": int(y.sum()),
           "n_cal_nodes": int(n_cal), "epoch_rule": epoch_rule, "norm": norm, "members": per_member}

    # --- instability of the single-run detector
    keys = ["auroc", "ap"]
    out["member_spread"] = {
        k: {"mean": float(np.nanmean([pm[k] or np.nan for pm in per_member])),
            "std": float(np.nanstd([pm[k] or np.nan for pm in per_member])),
            "min": float(np.nanmin([pm[k] or np.nan for pm in per_member])),
            "max": float(np.nanmax([pm[k] or np.nan for pm in per_member]))} for k in keys}
    out["member_spread"]["maxval_tp"] = [pm["maxval"]["tp"] for pm in per_member]
    out["member_spread"]["maxval_fp"] = [pm["maxval"]["fp"] for pm in per_member]

    # --- Q2: ensemble vs single members
    out["ensemble"] = {
        "S_auroc": _auc(y, S), "S_ap": _ap(y, S),
        "rawmean_auroc": _auc(y, np.vstack(raw_test).mean(0)),
        "rawmean_ap": _ap(y, np.vstack(raw_test).mean(0)),
        "vote_auroc": _auc(y, V),
        "maxcal": _confusion(y, S > S_cal.max()),
        "majority_vote": _confusion(y, V > 0.5),
        "at_alpha": {str(a): _confusion(y, p_ens <= a) for a in ALPHAS},
    }

    # --- Q1: does U add information beyond S?
    q1 = {"spearman_S_U": float(spearmanr(S, U).correlation)}
    X_s = S.reshape(-1, 1)
    X_su = np.column_stack([S, U])
    X_suv = np.column_stack([S, U, S * U])
    q1["cv_S"] = cv_auc(X_s, y)
    q1["cv_S_U"] = cv_auc(X_su, y)
    q1["cv_S_U_SxU"] = cv_auc(X_suv, y)
    q1["flagged"] = {}
    q1["near_band"] = {}
    for a in [1e-3, 1e-2]:
        f = p_ens <= a
        # AUROC of LOW disagreement for being a true positive among flagged nodes
        q1["flagged"][str(a)] = {"n": int(f.sum()), "tp": int(y[f].sum()),
                                 "auroc_negU_tp_vs_fp": _auc(y[f], -U[f]) if f.any() else None,
                                 "auroc_S_tp_vs_fp": _auc(y[f], S[f]) if f.any() else None}
        band = (p_ens > a) & (p_ens <= 10 * a)
        q1["near_band"][str(a)] = {"n": int(band.sum()), "fn": int(y[band].sum()),
                                   "auroc_U_fn_vs_tn": _auc(y[band], U[band]) if band.any() else None,
                                   "auroc_S_fn_vs_tn": _auc(y[band], S[band]) if band.any() else None}
    out["q1"] = q1

    # --- Q3: empirical FAR on benign test nodes vs nominal alpha
    benign = ~y
    out["q3_far"] = {"ensemble": conformal.empirical_far(p_ens[benign], ALPHAS)}
    # single member (seed of the first member), calibrated on its own B half
    m0 = members[0]
    ep0 = per_member[0]["epoch"]
    val0 = unit_table(data.read_split(m0["edge_losses_dir"], "val", ep0, cache_dir), unit, window)
    b0 = val0["score"].values[~_half_keys(val0.index)]
    p0 = conformal.pvalues(b0, raw_test[0])
    out["q3_far"]["member0"] = conformal.empirical_far(p0[benign], ALPHAS)

    arrays = {"S": S, "U": U, "V": V, "p_ens": p_ens, "y": y}
    if isinstance(test_index, pd.MultiIndex):
        arrays["w"] = test_index.get_level_values(0).values
        arrays["node"] = test_index.get_level_values(1).values
    else:
        arrays["key"] = test_index.values
    return out, arrays


def to_markdown(key, res):
    system, dataset, group = key
    L = [f"### {system} / {dataset} (group {group}) — unit={res['unit']} ({res['window']}), "
         f"M={res['n_members']}, {res['n_malicious_test_nodes']} malicious / {res['n_test_nodes']} "
         f"test units, {res['n_cal_nodes']} calibration units", ""]
    ms = res["member_spread"]
    L.append(f"- Single members: AUROC {ms['auroc']['mean']:.3f}±{ms['auroc']['std']:.3f} "
             f"[{ms['auroc']['min']:.3f}, {ms['auroc']['max']:.3f}], AP {ms['ap']['mean']:.3f}±{ms['ap']['std']:.3f} "
             f"[{ms['ap']['min']:.3f}, {ms['ap']['max']:.3f}]")
    L.append(f"- Max-val TP per member: {ms['maxval_tp']}; FP: {ms['maxval_fp']}")
    e = res["ensemble"]
    L.append(f"- Ensemble S: AUROC {e['S_auroc']}, AP {e['S_ap']} | raw-mean AP {e['rawmean_ap']} | "
             f"max-cal {e['maxcal']} | majority vote {e['majority_vote']}")
    q1 = res["q1"]
    L.append(f"- Q1 spearman(S,U)={q1['spearman_S_U']:.3f}; CV AP: S={q1['cv_S'] and q1['cv_S']['ap']}, "
             f"S+U={q1['cv_S_U'] and q1['cv_S_U']['ap']}, S+U+SxU={q1['cv_S_U_SxU'] and q1['cv_S_U_SxU']['ap']}")
    for a, v in q1["flagged"].items():
        L.append(f"  - flagged p<={a}: n={v['n']} tp={v['tp']} AUROC(-U: TP vs FP)={v['auroc_negU_tp_vs_fp']} "
                 f"AUROC(S)={v['auroc_S_tp_vs_fp']}")
    for a, v in q1["near_band"].items():
        L.append(f"  - band ({a}, {10*float(a):g}]: n={v['n']} fn={v['fn']} AUROC(U: FN vs TN)={v['auroc_U_fn_vs_tn']} "
                 f"AUROC(S)={v['auroc_S_fn_vs_tn']}")
    L.append("- Q3 empirical FAR (benign test nodes) vs alpha:")
    L.append("  | alpha | ensemble | member0 |")
    L.append("  |---|---|---|")
    for a in ALPHAS:
        L.append(f"  | {a:g} | {res['q3_far']['ensemble'][a]:.5f} | {res['q3_far']['member0'][a]:.5f} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system")
    ap.add_argument("--dataset")
    ap.add_argument("--group")
    ap.add_argument("--epoch_rule", default="min_val_loss", choices=["min_val_loss", "last"])
    ap.add_argument("--min_members", type=int, default=3)
    ap.add_argument("--norm", default="robust_z", choices=["robust_z", "logp"])
    ap.add_argument("--unit", default="node", choices=["node", "entity_window", "window"])
    ap.add_argument("--window", default="batch", help="'batch' or '<N>min', for time-based units")
    args = ap.parse_args()

    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    rep_dir = os.path.join(args.artifact_dir, "caliber", "reports")
    os.makedirs(rep_dir, exist_ok=True)
    groups = data.group_manifests(data.load_manifests(args.artifact_dir, args.system, args.dataset, args.group))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    md = [f"# CALIBER pilot report ({stamp} UTC, epoch rule: {args.epoch_rule})", ""]
    for key, members in sorted(groups.items()):
        if len(members) < args.min_members:
            print(f"skip {key}: only {len(members)} members")
            continue
        print(f"== {key}: {len(members)} members")
        labels, _ = data.load_labels(args.artifact_dir, key[1], members[0]["ground_truth_version"])
        res, arrays = analyse_group(members, labels, args.epoch_rule, cache, norm=args.norm,
                                    unit=args.unit, window=args.window)
        tag = f"{args.unit}-{args.window}_{args.epoch_rule}_{args.norm}"
        base = os.path.join(rep_dir, f"pilot_{key[0]}_{key[1]}_{key[2]}_{tag}")
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        pd.DataFrame(arrays).to_parquet(base + "_units.parquet", index=False)
        md.append(to_markdown(key, res))
        print(md[-1])
    out = os.path.join(rep_dir, f"pilot_{stamp}_{args.unit}-{args.window}_{args.epoch_rule}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
