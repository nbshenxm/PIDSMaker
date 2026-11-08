"""Margin-to-threshold analysis for Velox node-level evaluation output.

Answers: "if we trust the node-eval threshold as-is, does a node's distance
from it correlate with whether the prediction is actually correct?"

Works directly off the artifacts the evaluation pipeline already writes to
`<eval_dir>/precision_recall_dir/scores_model_epoch_<N>.pkl` (a torch.save'd
dict with pred_scores / y_preds / y_truth / nodes). The decision threshold is
not read from a config or log line -- it is inferred straight from the data
(max score among nodes predicted negative), which matches the logged
threshold to ~1e-6 in practice, so this script has no hardcoded per-run
constants and works on any future run unmodified.

Usage:
    python -m pidsmaker.pids.velox.margin_analysis --scores-dir \
        ./artifacts/evaluation/evaluation/<hash>/<DATASET>/precision_recall_dir \
        --epochs 0 1 2 3 4 5 6 7 8 9 10 11

    # or let it auto-discover every scores_model_epoch_*.pkl in the dir:
    python -m pidsmaker.pids.velox.margin_analysis --scores-dir <dir> --epochs all

See MARGIN_ANALYSIS_FINDINGS.md in this folder for the conclusions drawn from
running this against the CADETS_E3 / velox run (slurm job 39266).
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

DEFAULT_BAND_EDGES = [0, -0.5, -1, -1.5, -2, -3, -4, -6, -8, -12]


def load_epoch_scores(scores_dir, epoch):
    path = os.path.join(scores_dir, f"scores_model_epoch_{epoch}.pkl")
    d = torch.load(path, weights_only=False)
    return np.array(d["pred_scores"]), np.array(d["y_truth"]), np.array(d["y_preds"])


def infer_threshold(scores, y_preds):
    """Recover the decision threshold from data instead of requiring it as input.

    node_evaluation.py sets y_hat = int(pred_score > thr), so the largest
    score among y_hat==0 nodes is a tight lower-bound proxy for thr (matches
    the true value to ~1e-6 in practice, since ties at the boundary are rare
    once threshold is an actual val-set score).
    """
    below = scores[y_preds == 0]
    if len(below) == 0:
        raise ValueError("No y_preds==0 nodes found; cannot infer threshold.")
    return float(below.max())


def margin_auc_summary(scores, y_truth, y_preds, thr):
    margin = scores - thr
    above = y_preds == 1
    below = ~above

    above_is_tp = y_truth[above]
    n_tp, n_fp = int(above_is_tp.sum()), int((1 - above_is_tp).sum())
    auc_above = (
        roc_auc_score(above_is_tp, margin[above]) if n_tp > 0 and n_fp > 0 else float("nan")
    )

    below_is_tn = 1 - y_truth[below]
    n_tn, n_fn = int(below_is_tn.sum()), int((1 - below_is_tn).sum())
    auc_below = (
        roc_auc_score(below_is_tn, margin[below]) if n_tn > 0 and n_fn > 0 else float("nan")
    )

    return {
        "thr": thr,
        "n_above": int(above.sum()),
        "n_tp": n_tp,
        "n_fp": n_fp,
        "auc_margin_above_tp_vs_fp": auc_above,
        "n_below": int(below.sum()),
        "n_tn": n_tn,
        "n_fn": n_fn,
        "auc_margin_below_tn_vs_fn": auc_below,
    }


def margin_band_table(scores, y_truth, y_preds, thr, band_edges=DEFAULT_BAND_EDGES):
    """Bin below-threshold nodes by distance below thr and report, per band,
    how many nodes fall in it and how many are actually malicious (missed
    attacks reachable by widening the cutoff into that band)."""
    margin = scores - thr
    below = y_preds == 0
    total_malicious = int(y_truth.sum())

    rows = []
    cum_fn, cum_n = 0, 0
    for hi, lo in zip(band_edges[:-1], band_edges[1:]):
        mask = below & (margin > lo) & (margin <= hi)
        n = int(mask.sum())
        n_fn = int(y_truth[mask].sum())
        cum_fn += n_fn
        cum_n += n
        rows.append(
            {
                "band_lo": lo,
                "band_hi": hi,
                "n_nodes": n,
                "n_fn_in_band": n_fn,
                "cum_fn": cum_fn,
                "cum_recall_if_included": cum_fn / total_malicious if total_malicious else 0.0,
                "cum_extra_nodes_to_review": cum_n,
            }
        )
    return rows


def discover_epochs(scores_dir):
    paths = glob.glob(os.path.join(scores_dir, "scores_model_epoch_*.pkl"))
    epochs = []
    for p in paths:
        m = re.search(r"scores_model_epoch_(\d+)\.pkl$", p)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-dir",
        required=True,
        help="Path to the evaluation run's precision_recall_dir containing scores_model_epoch_*.pkl",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        default=["all"],
        help="Epoch numbers to analyze, or 'all' to auto-discover every scores_model_epoch_*.pkl",
    )
    parser.add_argument(
        "--bands",
        action="store_true",
        help="Also print the margin-band table (recall reachable by widening the cutoff)",
    )
    args = parser.parse_args()

    epochs = (
        discover_epochs(args.scores_dir)
        if args.epochs == ["all"]
        else [int(e) for e in args.epochs]
    )

    print(
        f"{'ep':>3} {'thr':>10} | {'n_abv':>6} {'tp':>3} {'fp':>4} {'AUC(margin->TP|above)':>23} | "
        f"{'n_blw':>7} {'fn':>3} {'tn':>7} {'AUC(margin->TN|below)':>23}"
    )
    for ep in epochs:
        scores, y_truth, y_preds = load_epoch_scores(args.scores_dir, ep)
        thr = infer_threshold(scores, y_preds)
        s = margin_auc_summary(scores, y_truth, y_preds, thr)
        print(
            f"{ep:>3} {s['thr']:>10.4f} | {s['n_above']:>6} {s['n_tp']:>3} {s['n_fp']:>4} "
            f"{s['auc_margin_above_tp_vs_fp']:>23.3f} | "
            f"{s['n_below']:>7} {s['n_fn']:>3} {s['n_tn']:>7} {s['auc_margin_below_tn_vs_fn']:>23.3f}"
        )

        if args.bands:
            rows = margin_band_table(scores, y_truth, y_preds, thr)
            print(f"      threshold={thr:.4f}, bands below it:")
            for r in rows:
                print(
                    f"        ({r['band_lo']:>6.1f}, {r['band_hi']:>5.1f}] n={r['n_nodes']:>8} "
                    f"fn_here={r['n_fn_in_band']:>4} cum_recall={r['cum_recall_if_included']:>6.1%} "
                    f"cum_nodes_to_review={r['cum_extra_nodes_to_review']:>8}"
                )


if __name__ == "__main__":
    main()
