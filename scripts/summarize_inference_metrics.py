#!/usr/bin/env python3
"""Summarize interpretable detection metrics from cross-dataset inference outputs.

This script reads artifacts/inference_results.json, locates saved edge-loss CSVs for each
successful evaluation, reconstructs labels from ground truth, and reports:
- TP, FP, TN, FN
- precision, recall, fscore
- AP, AUC, accuracy, balanced_acc, MCC

By default it mirrors edge_evaluation threshold behavior. If per-dataset validation losses
are missing, it falls back to using test losses as threshold reference and records that.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.detection.evaluation_methods.edge_evaluation import get_edge_predictions
from pidsmaker.detection.evaluation_methods.evaluation_utils import classifier_evaluation


def to_json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalar values to plain Python JSON-safe values."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def load_results(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload.setdefault("results", [])
        return payload
    if isinstance(payload, list):
        return {"results": payload}
    raise ValueError(f"Unsupported results format in {path}")


def build_cfg(model: str, dataset: str):
    args = get_runtime_required_args(
        args=[
            model,
            dataset,
            "--artifact_dir",
            "./artifacts/",
            "--database_host",
            "localhost",
        ]
    )
    return get_yml_cfg(args)


def find_test_dir(dataset: str, epoch: int) -> Path | None:
    pattern = f"artifacts/training/training/*/{dataset}/edge_losses/test/model_epoch_{epoch}"
    matches = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def summarize_one(row: Dict[str, Any]) -> Dict[str, Any]:
    system = row.get("detection_system")
    dataset = row.get("testing_dataset")
    epoch = row.get("loaded_epoch")

    out = {
        "detection_system": system,
        "training_dataset": row.get("training_dataset"),
        "testing_dataset": dataset,
        "run_hash": row.get("run_hash"),
        "status": row.get("status"),
        "success": bool(row.get("success")),
    }

    if not out["success"]:
        out["reason"] = row.get("db_reason") or row.get("message")
        return out

    if epoch is None:
        out["reason"] = "missing_loaded_epoch"
        return out

    test_dir = find_test_dir(dataset, int(epoch))
    if test_dir is None:
        out["reason"] = f"edge_losses_not_found_for_epoch_{epoch}"
        return out

    val_dir = test_dir.parent.parent / "val" / f"model_epoch_{epoch}"
    threshold_source = "val"
    if not val_dir.exists():
        val_dir = test_dir
        threshold_source = "test_fallback"

    cfg = build_cfg(system, dataset)
    scores, y_true, y_hat, _src_dst_t_type, thr = get_edge_predictions(
        str(val_dir), str(test_dir), cfg
    )
    stats = classifier_evaluation(y_true, y_hat, scores)

    out.update({k: to_json_safe(v) for k, v in stats.items()})
    out["threshold"] = float(thr)
    out["threshold_source"] = threshold_source
    out["num_scored_edges"] = len(scores)
    out["num_positive_edges"] = int(sum(y_true))
    return out


def print_table(rows: List[Dict[str, Any]]) -> None:
    print(
        f"{'Dataset':<18} {'TP':>8} {'FP':>10} {'FN':>8} {'TN':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} {'AP':>8}"
    )
    print("-" * 112)
    for r in rows:
        if "tp" not in r:
            print(f"{r.get('testing_dataset', 'unknown'):<18} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue
        print(
            f"{r['testing_dataset']:<18} {r['tp']:>8} {r['fp']:>10} {r['fn']:>8} {r['tn']:>10} "
            f"{r['precision']:>10.5f} {r['recall']:>8.5f} {r['fscore']:>8.5f} {r['auc']:>8.5f} {r['ap']:>8.5f}"
        )



def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TP/FP/recall/precision from inference outputs")
    parser.add_argument(
        "--results-file",
        default="artifacts/inference_results.json",
        help="Path to inference results JSON",
    )
    parser.add_argument("--detection-system", default=None)
    parser.add_argument("--testing-dataset", default=None)
    parser.add_argument(
        "--export-json",
        default="artifacts/evaluation/evaluation/inference_interpretable_metrics.json",
        help="Path to write summarized metrics JSON",
    )
    args = parser.parse_args()

    payload = load_results(Path(args.results_file))
    rows = payload.get("results", [])

    if args.detection_system:
        rows = [r for r in rows if r.get("detection_system") == args.detection_system]
    if args.testing_dataset:
        rows = [r for r in rows if r.get("testing_dataset") == args.testing_dataset]

    summaries = [summarize_one(r) for r in rows]

    print_table(summaries)

    out_path = Path(args.export_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries}, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
