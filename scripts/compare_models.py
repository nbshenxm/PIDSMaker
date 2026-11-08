#!/usr/bin/env python3
"""Compare models by mining cross-dataset inference results.

Expected input format:
- artifacts/inference_results.json produced by scripts/evaluate_all_models.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List


def load_results(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Backward compatibility with old list-only format
    if isinstance(payload, list):
        return {
            "schema_version": 1,
            "results": payload,
            "selection": {},
            "summary": {},
        }
    if isinstance(payload, dict):
        payload.setdefault("results", [])
        payload.setdefault("selection", {})
        payload.setdefault("summary", {})
        return payload
    raise ValueError("Unsupported results file structure")


def model_key(row: Dict) -> tuple:
    return (
        row.get("detection_system") or row.get("system") or "unknown",
        row.get("training_dataset") or row.get("dataset") or "unknown",
        row.get("run_hash") or "unknown",
    )


def get_test_loss(row: Dict):
    metrics = row.get("metrics") or {}
    value = metrics.get("test_loss")
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def aggregate_by_model(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[model_key(r)].append(r)

    summary = []
    for (system, train_dataset, run_hash), group in grouped.items():
        successes = [g for g in group if g.get("success")]
        losses = [x for x in (get_test_loss(g) for g in successes) if x is not None]
        success_rate = len(successes) / len(group) if group else 0.0

        summary.append(
            {
                "detection_system": system,
                "training_dataset": train_dataset,
                "run_hash": run_hash,
                "eval_count": len(group),
                "success_count": len(successes),
                "success_rate": success_rate,
                "mean_test_loss": mean(losses) if losses else None,
                "min_test_loss": min(losses) if losses else None,
                "max_test_loss": max(losses) if losses else None,
                "selected_model_best_validation_score": group[0].get(
                    "selected_model_best_validation_score"
                ),
                "seed": group[0].get("seed"),
            }
        )

    summary.sort(
        key=lambda x: (
            x["mean_test_loss"] if x["mean_test_loss"] is not None else float("inf"),
            -x["success_rate"],
        )
    )
    return summary


def best_model_per_test_dataset(rows: List[Dict]) -> List[Dict]:
    by_test_dataset = defaultdict(list)
    for row in rows:
        if row.get("success"):
            by_test_dataset[row.get("testing_dataset", row.get("dataset", "unknown"))].append(row)

    winners = []
    for test_dataset, group in sorted(by_test_dataset.items()):
        scored = []
        for row in group:
            tl = get_test_loss(row)
            if tl is not None:
                scored.append((tl, row))
        if not scored:
            continue

        scored.sort(key=lambda x: x[0])
        best_loss, best_row = scored[0]
        winners.append(
            {
                "testing_dataset": test_dataset,
                "detection_system": best_row.get("detection_system", best_row.get("system")),
                "training_dataset": best_row.get("training_dataset", best_row.get("dataset")),
                "run_hash": best_row.get("run_hash"),
                "test_loss": best_loss,
            }
        )
    return winners


def print_model_ranking(ranking: List[Dict], top_k: int):
    print("\n" + "=" * 90)
    print("Model Ranking by Mean Test Loss (lower is better)")
    print("=" * 90)
    print(
        f"{'Rank':<5} {'System':<12} {'Train Dataset':<15} {'Mean Test Loss':<15} {'Success':<12} {'Seed':<8} {'Run Hash':<20}"
    )
    print("-" * 90)

    for idx, row in enumerate(ranking[:top_k], start=1):
        mean_loss = f"{row['mean_test_loss']:.6f}" if row["mean_test_loss"] is not None else "N/A"
        success = f"{row['success_count']}/{row['eval_count']}"
        print(
            f"{idx:<5} {row['detection_system']:<12} {row['training_dataset']:<15} "
            f"{mean_loss:<15} {success:<12} {str(row['seed']):<8} {row['run_hash'][:20]:<20}"
        )


def print_dataset_winners(winners: List[Dict]):
    print("\n" + "=" * 90)
    print("Best Model per Testing Dataset (by test_loss)")
    print("=" * 90)
    print(f"{'Test Dataset':<15} {'System':<12} {'Train Dataset':<15} {'Test Loss':<12} {'Run Hash':<20}")
    print("-" * 90)
    for w in winners:
        print(
            f"{w['testing_dataset']:<15} {w['detection_system']:<12} {w['training_dataset']:<15} "
            f"{w['test_loss']:<12.6f} {str(w['run_hash'])[:20]:<20}"
        )


def main():
    parser = argparse.ArgumentParser(description="Mine cross-dataset model performance results.")
    parser.add_argument(
        "--results-file",
        default="artifacts/inference_results.json",
        help="Path to evaluation results JSON from scripts/evaluate_all_models.py",
    )
    parser.add_argument("--detection-system", default=None)
    parser.add_argument("--training-dataset", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--export-summary",
        default="",
        help="Optional output path for mined summary JSON.",
    )
    args = parser.parse_args()

    payload = load_results(Path(args.results_file))
    rows = payload.get("results", [])

    # Backward compatibility: old rows used keys system/dataset and no metrics.
    if not rows:
        print("No results to analyze.")
        return

    if args.detection_system:
        rows = [
            r
            for r in rows
            if (r.get("detection_system") or r.get("system")) == args.detection_system
        ]
    if args.training_dataset:
        rows = [
            r
            for r in rows
            if (r.get("training_dataset") or r.get("dataset")) == args.training_dataset
        ]

    if not rows:
        print("No rows matched the requested filters.")
        return

    ranking = aggregate_by_model(rows)
    winners = best_model_per_test_dataset(rows)

    print_model_ranking(ranking, top_k=args.top_k)
    print_dataset_winners(winners)

    overall_best = ranking[0] if ranking else None
    if overall_best:
        print("\n" + "=" * 90)
        print("Recommended Overall Model")
        print("=" * 90)
        print(f"Detection system: {overall_best['detection_system']}")
        print(f"Training dataset: {overall_best['training_dataset']}")
        print(f"Run hash: {overall_best['run_hash']}")
        print(f"Mean test loss: {overall_best['mean_test_loss']}")
        print(f"Success rate: {overall_best['success_count']}/{overall_best['eval_count']}")

    if args.export_summary:
        out = {
            "schema_version": 1,
            "source_results_file": args.results_file,
            "model_ranking": ranking,
            "dataset_winners": winners,
            "overall_best": overall_best,
        }
        out_path = Path(args.export_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nSaved mined summary to {out_path}")


if __name__ == "__main__":
    main()
