#!/usr/bin/env python3
"""Evaluate selected trained models across test datasets.

Selection rule:
- Discover all trained runs under artifacts/training/training
- Group by (detection_system, training_dataset)
- Keep only one model per group: highest checkpoint best validation score

Then evaluate each selected model on all requested test datasets and save results to JSON.
"""

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2

from pidsmaker.config import DATASET_DEFAULT_CONFIG
from pidsmaker.config import get_runtime_required_args, get_yml_cfg


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def to_float_or_none(value):
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def parse_int_or_none(value):
    try:
        return int(value)
    except Exception:
        return None


def list_available_test_datasets() -> List[str]:
    available = list(DATASET_DEFAULT_CONFIG.keys())
    lower_to_dataset = {name.lower(): name for name in available}

    data_dir = Path("data")
    datasets = set()
    if not data_dir.exists():
        return []

    for file in data_dir.iterdir():
        name = file.name
        if name.endswith(".dump"):
            key = name[:-5]
            canonical = lower_to_dataset.get(key.lower())
            if canonical is not None:
                datasets.add(canonical)
    return sorted(datasets)


def normalize_dataset_name(name: str) -> Optional[str]:
    available = list(DATASET_DEFAULT_CONFIG.keys())
    lower_to_dataset = {dataset.lower(): dataset for dataset in available}
    return lower_to_dataset.get(name.lower())


def discover_model_candidates(artifacts_dir: Path) -> List[Dict]:
    candidates: List[Dict] = []
    training_root = artifacts_dir / "training" / "training"
    if not training_root.exists():
        return candidates

    for run_hash_dir in sorted(training_root.iterdir()):
        if not run_hash_dir.is_dir():
            continue

        for train_dataset_dir in sorted(run_hash_dir.iterdir()):
            if not train_dataset_dir.is_dir():
                continue

            checkpoint_dir = train_dataset_dir / "checkpoints"
            if not checkpoint_dir.exists():
                continue

            checkpoints_meta = load_json(checkpoint_dir / "checkpoints.json")
            run_meta = load_json(train_dataset_dir / "run_metadata.json")

            detection_system = run_meta.get("detection_system")
            train_dataset = train_dataset_dir.name
            best_score = to_float_or_none(
                checkpoints_meta.get("best_val_score", checkpoints_meta.get("best_validation_score"))
            )
            best_epoch = parse_int_or_none(checkpoints_meta.get("best_checkpoint"))
            seed = run_meta.get("effective_training_seed", run_meta.get("input_train_seed"))

            ckpt_meta_file = checkpoint_dir / "checkpoints.json"
            mtime = ckpt_meta_file.stat().st_mtime if ckpt_meta_file.exists() else checkpoint_dir.stat().st_mtime

            candidates.append(
                {
                    "detection_system": detection_system,
                    "training_dataset": train_dataset,
                    "run_hash": run_hash_dir.name,
                    "checkpoint_dir": str(checkpoint_dir),
                    "best_validation_score": best_score,
                    "best_epoch": best_epoch,
                    "seed": seed,
                    "has_run_metadata": bool(run_meta),
                    "metadata_created_at_utc": run_meta.get("created_at_utc"),
                    "selection_mtime": mtime,
                }
            )

    return candidates


def choose_best_models(candidates: List[Dict], include_unknown_systems: bool = False) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    skipped_unknown = []

    for c in candidates:
        system = c.get("detection_system")
        if not system:
            if include_unknown_systems:
                system = "unknown"
                c = dict(c)
                c["detection_system"] = system
            else:
                skipped_unknown.append(c)
                continue
        grouped[(system, c["training_dataset"])].append(c)

    selected = []
    dropped = []

    for key, group in grouped.items():
        best = max(
            group,
            key=lambda x: (
                x["best_validation_score"] if x["best_validation_score"] is not None else float("-inf"),
                x["selection_mtime"],
            ),
        )
        selected.append(best)
        for item in group:
            if item is not best:
                dropped.append(item)

    return {
        "selected": selected,
        "dropped_duplicates": dropped,
        "skipped_unknown": skipped_unknown,
    }


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def run_inference_once(
    model: Dict,
    test_dataset: str,
    epoch: int,
    metrics_dir: Path,
    timeout: int,
) -> Dict:
    system = model["detection_system"]
    train_dataset = model["training_dataset"]
    run_hash = model["run_hash"]

    out_name = safe_name(f"{system}__train_{train_dataset}__test_{test_dataset}__{run_hash}.json")
    metrics_file = metrics_dir / out_name

    cmd = [
        "python",
        "-m",
        "pidsmaker.detection.inference",
        f"--config={system}",
        f"--dataset={test_dataset}",
        f"--checkpoint-dir={model['checkpoint_dir']}",
        "--split=test",
        "--no-wandb",
        f"--epoch={epoch}",
        f"--metrics-out={metrics_file}",
    ]

    run_kwargs = {
        "capture_output": True,
        "text": True,
    }
    if timeout > 0:
        run_kwargs["timeout"] = timeout

    result = subprocess.run(cmd, **run_kwargs)
    metrics_payload = load_json(metrics_file)

    if result.returncode == 0:
        return {
            "success": True,
            "message": "inference completed",
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1200:],
            "stderr_tail": result.stderr[-1200:],
            "metrics_file": str(metrics_file),
            "metrics": metrics_payload.get("stats", {}),
            "loaded_epoch": metrics_payload.get("epoch"),
        }

    err_text = (result.stderr or result.stdout or "").strip()
    return {
        "success": False,
        "message": err_text[:400],
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
        "metrics_file": str(metrics_file),
        "metrics": metrics_payload.get("stats", {}),
        "loaded_epoch": metrics_payload.get("epoch"),
    }


def get_cfg_for_system_dataset(system: str, dataset: str):
    runtime_args = get_runtime_required_args(
        args=[
            system,
            dataset,
            "--artifact_dir",
            "./artifacts/",
            "--database_host",
            "localhost",
        ]
    )
    return get_yml_cfg(runtime_args)


def get_missing_required_artifacts(system: str, dataset: str) -> Dict:
    try:
        cfg = get_cfg_for_system_dataset(system, dataset)
    except Exception as e:
        return {"ok": False, "missing": [], "error": str(e)}

    required_paths = [
        os.path.join(cfg.feat_inference._edge_embeds_dir, "train"),
        os.path.join(cfg.feat_inference._edge_embeds_dir, "val"),
        os.path.join(cfg.feat_inference._edge_embeds_dir, "test"),
    ]

    # Batching on-disk artifacts are required only when save_on_disk is enabled.
    # If save_on_disk=False, batching can run in-memory during inference.
    if cfg.batching.save_on_disk:
        required_paths.append(cfg.batching._preprocessed_graphs_dir)

    missing = [p for p in required_paths if not os.path.isdir(p)]
    return {
        "ok": True,
        "missing": missing,
        "error": None,
        "batching_save_on_disk": bool(cfg.batching.save_on_disk),
    }


def build_prep_command(system: str, dataset: str) -> str:
    return (
        "python scripts/prepare_inference_artifacts.py "
        f"{system} {dataset} "
        "--artifact-dir ./artifacts/ --database-host localhost"
    )


def run_prepare_command(system: str, dataset: str, timeout: int = 7200) -> Dict:
    cmd = [
        "python",
        "scripts/prepare_inference_artifacts.py",
        system,
        dataset,
        "--artifact-dir",
        "./artifacts/",
        "--database-host",
        "localhost",
    ]
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = deque(maxlen=500)
        start = time.monotonic()

        assert process.stdout is not None
        while True:
            line = process.stdout.readline()
            if line:
                output_lines.append(line)
                print(f"    [prep] {line.rstrip()}")

            if process.poll() is not None:
                for trailing in process.stdout:
                    output_lines.append(trailing)
                    print(f"    [prep] {trailing.rstrip()}")
                break

            if timeout > 0 and (time.monotonic() - start > timeout):
                process.kill()
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
                out_text = "".join(output_lines)
                return {
                    "success": False,
                    "command": " ".join(shlex.quote(x) for x in cmd),
                    "returncode": None,
                    "stdout_tail": out_text[-1200:],
                    "stderr_tail": "",
                    "error": f"prep timeout after {timeout}s",
                }

        out_text = "".join(output_lines)
        return {
            "success": process.returncode == 0,
            "command": " ".join(shlex.quote(x) for x in cmd),
            "returncode": process.returncode,
            "stdout_tail": out_text[-1200:],
            "stderr_tail": "",
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "command": " ".join(shlex.quote(x) for x in cmd),
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select one best model per (detection system, training dataset) and evaluate on test datasets."
        )
    )
    parser.add_argument("--detection-system", default=None)
    parser.add_argument("--training-dataset", default=None)
    parser.add_argument(
        "--test-dataset",
        default=None,
        help="Evaluate only one target test dataset. Omit to evaluate on all available datasets.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="Checkpoint epoch passed to inference (-1 means best checkpoint).",
    )
    parser.add_argument("--list-models-only", action="store_true")
    parser.add_argument("--include-unknown-systems", action="store_true")
    parser.add_argument("--results-file", default="artifacts/inference_results.json")
    parser.add_argument(
        "--no-skip-missing-artifacts",
        action="store_true",
        help="If set, still attempt inference even when required artifacts are missing.",
    )
    parser.add_argument(
        "--auto-prepare-missing",
        dest="auto_prepare_missing",
        action="store_true",
        default=True,
        help="Automatically run prep script when required artifacts are missing (default: enabled).",
    )
    parser.add_argument(
        "--no-auto-prepare-missing",
        dest="auto_prepare_missing",
        action="store_false",
        help="Disable automatic prep when required artifacts are missing.",
    )
    parser.add_argument(
        "--prepare-timeout-seconds",
        type=int,
        default=0,
        help="Timeout for each auto-prepare command. Use 0 or negative for no timeout.",
    )
    parser.add_argument(
        "--inference-timeout-seconds",
        type=int,
        default=0,
        help="Timeout for each inference command. Use 0 or negative for no timeout.",
    )
    args = parser.parse_args()

    artifacts_dir = Path("artifacts")
    candidates = discover_model_candidates(artifacts_dir)
    if not candidates:
        print("No checkpoint candidates found under artifacts/training/training.")
        return

    if args.detection_system:
        candidates = [c for c in candidates if c.get("detection_system") == args.detection_system]
    if args.training_dataset:
        candidates = [c for c in candidates if c.get("training_dataset") == args.training_dataset]

    if not candidates:
        print("No candidates matched requested filters.")
        return

    selection = choose_best_models(candidates, include_unknown_systems=args.include_unknown_systems)
    selected = sorted(
        selection["selected"],
        key=lambda x: (x["detection_system"], x["training_dataset"]),
    )

    print("\nSelected models (one per detection system + training dataset):")
    for m in selected:
        print(
            f"  - system={m['detection_system']}, train_dataset={m['training_dataset']}, "
            f"seed={m['seed']}, best_val={m['best_validation_score']}, run_hash={m['run_hash']}"
        )

    if selection["dropped_duplicates"]:
        print(f"\nDropped duplicate runs: {len(selection['dropped_duplicates'])}")
    if selection["skipped_unknown"]:
        print(f"Skipped runs without run_metadata detection_system: {len(selection['skipped_unknown'])}")

    if args.list_models_only:
        return

    if args.test_dataset:
        canonical = normalize_dataset_name(args.test_dataset)
        if canonical is None:
            print(
                f"Unknown test dataset: {args.test_dataset}. "
                f"Available: {sorted(DATASET_DEFAULT_CONFIG.keys())}"
            )
            return
        test_datasets = [canonical]
    else:
        test_datasets = list_available_test_datasets()

    if not test_datasets:
        print("No test datasets found in data directory.")
        return

    metrics_dir = artifacts_dir / "evaluation" / "evaluation" / "cross_dataset_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(selected) * len(test_datasets)
    done = 0
    prep_commands = {}

    for model in selected:
        for test_dataset in test_datasets:
            done += 1
            system = model["detection_system"]
            print(
                f"[{done}/{total}] Evaluating {system} "
                f"trained_on={model['training_dataset']} tested_on={test_dataset}"
            )

            preflight = get_missing_required_artifacts(system, test_dataset)
            if not preflight["ok"]:
                message = "Skipped: could not compute required artifact paths for dataset/model pair."
                row = {
                    "detection_system": system,
                    "training_dataset": model["training_dataset"],
                    "testing_dataset": test_dataset,
                    "run_hash": model["run_hash"],
                    "checkpoint_dir": model["checkpoint_dir"],
                    "selected_model_best_validation_score": model["best_validation_score"],
                    "selected_model_best_epoch": model["best_epoch"],
                    "seed": model["seed"],
                    "success": False,
                    "status": "skipped_preflight_error",
                    "message": message,
                    "preflight_error": preflight["error"],
                    "missing_artifacts": [],
                    "prep_command": build_prep_command(system, test_dataset),
                    "command": None,
                    "returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "metrics_file": "",
                    "metrics": {},
                    "loaded_epoch": None,
                }
                results.append(row)
                print(f"  -> SKIP: {message}")
                print(f"     preflight_error: {preflight['error']}")
                continue

            missing_paths = preflight["missing"]
            if missing_paths and args.auto_prepare_missing:
                try:
                    cfg = get_cfg_for_system_dataset(system, test_dataset)
                    target_db = (
                        cfg.dataset.database_all_file if cfg.construction.use_all_files else cfg.dataset.database
                    )

                    admin_conn = psycopg2.connect(
                        database="postgres",
                        host=cfg.database.host,
                        user=cfg.database.user,
                        password=cfg.database.password,
                        port=cfg.database.port,
                    )
                    admin_conn.autocommit = True
                    with admin_conn.cursor() as cur:
                        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
                        db_exists = cur.fetchone() is not None
                    admin_conn.close()

                    if not db_exists:
                        message = "Skipped: database preflight failed (database does not exist)."
                        row = {
                            "detection_system": system,
                            "training_dataset": model["training_dataset"],
                            "testing_dataset": test_dataset,
                            "run_hash": model["run_hash"],
                            "checkpoint_dir": model["checkpoint_dir"],
                            "selected_model_best_validation_score": model["best_validation_score"],
                            "selected_model_best_epoch": model["best_epoch"],
                            "seed": model["seed"],
                            "success": False,
                            "status": "skipped_db_sanity_failed",
                            "message": message,
                            "missing_artifacts": missing_paths,
                            "db_name": target_db,
                            "db_reason": "database_missing",
                            "db_fix_hint": "Run: (cd scripts/apptainer && make load-dumps)",
                            "prep_command": build_prep_command(system, test_dataset),
                            "command": None,
                            "returncode": None,
                            "stdout_tail": "",
                            "stderr_tail": "",
                            "metrics_file": "",
                            "metrics": {},
                            "loaded_epoch": None,
                        }
                        results.append(row)
                        print(f"  -> SKIP: {message} db={target_db}")
                        continue

                    target_conn = psycopg2.connect(
                        database=target_db,
                        host=cfg.database.host,
                        user=cfg.database.user,
                        password=cfg.database.password,
                        port=cfg.database.port,
                    )
                    with target_conn.cursor() as cur:
                        cur.execute("SELECT to_regclass('public.event_table')")
                        has_event_table = cur.fetchone()[0] is not None
                        event_count = 0
                        if has_event_table:
                            cur.execute("SELECT COUNT(*) FROM event_table")
                            event_count = int(cur.fetchone()[0] or 0)
                    target_conn.close()

                    if (not has_event_table) or event_count <= 0:
                        reason = "event_table_missing" if not has_event_table else "event_table_empty"
                        message = "Skipped: database preflight failed (no event data available)."
                        row = {
                            "detection_system": system,
                            "training_dataset": model["training_dataset"],
                            "testing_dataset": test_dataset,
                            "run_hash": model["run_hash"],
                            "checkpoint_dir": model["checkpoint_dir"],
                            "selected_model_best_validation_score": model["best_validation_score"],
                            "selected_model_best_epoch": model["best_epoch"],
                            "seed": model["seed"],
                            "success": False,
                            "status": "skipped_db_sanity_failed",
                            "message": message,
                            "missing_artifacts": missing_paths,
                            "db_name": target_db,
                            "db_reason": reason,
                            "db_event_count": event_count,
                            "db_fix_hint": "Run: (cd scripts/apptainer && make load-dumps)",
                            "prep_command": build_prep_command(system, test_dataset),
                            "command": None,
                            "returncode": None,
                            "stdout_tail": "",
                            "stderr_tail": "",
                            "metrics_file": "",
                            "metrics": {},
                            "loaded_epoch": None,
                        }
                        results.append(row)
                        print(
                            f"  -> SKIP: {message} db={target_db} reason={reason} event_count={event_count}"
                        )
                        continue
                except Exception as e:
                    message = "Skipped: database preflight failed (connection/query error)."
                    row = {
                        "detection_system": system,
                        "training_dataset": model["training_dataset"],
                        "testing_dataset": test_dataset,
                        "run_hash": model["run_hash"],
                        "checkpoint_dir": model["checkpoint_dir"],
                        "selected_model_best_validation_score": model["best_validation_score"],
                        "selected_model_best_epoch": model["best_epoch"],
                        "seed": model["seed"],
                        "success": False,
                        "status": "skipped_db_sanity_failed",
                        "message": message,
                        "missing_artifacts": missing_paths,
                        "db_reason": "db_connection_or_query_error",
                        "db_error": str(e),
                        "db_fix_hint": "Check postgres status, then run: (cd scripts/apptainer && make load-dumps)",
                        "prep_command": build_prep_command(system, test_dataset),
                        "command": None,
                        "returncode": None,
                        "stdout_tail": "",
                        "stderr_tail": "",
                        "metrics_file": "",
                        "metrics": {},
                        "loaded_epoch": None,
                    }
                    results.append(row)
                    print(f"  -> SKIP: {message} error={e}")
                    continue

                print("  -> AUTO-PREP: missing artifacts detected, preparing prerequisites...")
                prep_result = run_prepare_command(
                    system,
                    test_dataset,
                    timeout=args.prepare_timeout_seconds,
                )

                if prep_result["success"]:
                    post_prep = get_missing_required_artifacts(system, test_dataset)
                    if post_prep["ok"]:
                        missing_paths = post_prep["missing"]
                    else:
                        missing_paths = missing_paths
                else:
                    key = f"{system}::{test_dataset}"
                    prep_commands[key] = build_prep_command(system, test_dataset)
                    message = "Skipped: auto-prepare failed; see prep stderr/stdout tails."
                    row = {
                        "detection_system": system,
                        "training_dataset": model["training_dataset"],
                        "testing_dataset": test_dataset,
                        "run_hash": model["run_hash"],
                        "checkpoint_dir": model["checkpoint_dir"],
                        "selected_model_best_validation_score": model["best_validation_score"],
                        "selected_model_best_epoch": model["best_epoch"],
                        "seed": model["seed"],
                        "success": False,
                        "status": "skipped_prepare_failed",
                        "message": message,
                        "missing_artifacts": missing_paths,
                        "prep_command": prep_result["command"],
                        "prep_error": prep_result["error"],
                        "prep_stdout_tail": prep_result["stdout_tail"],
                        "prep_stderr_tail": prep_result["stderr_tail"],
                        "command": None,
                        "returncode": None,
                        "stdout_tail": "",
                        "stderr_tail": "",
                        "metrics_file": "",
                        "metrics": {},
                        "loaded_epoch": None,
                    }
                    results.append(row)
                    print(f"  -> SKIP: {message}")
                    continue

            if missing_paths and not args.no_skip_missing_artifacts:
                key = f"{system}::{test_dataset}"
                prep_commands[key] = build_prep_command(system, test_dataset)
                message = (
                    "Skipped: missing required artifacts for target dataset. "
                    "Run prep command listed in output payload."
                )
                row = {
                    "detection_system": system,
                    "training_dataset": model["training_dataset"],
                    "testing_dataset": test_dataset,
                    "run_hash": model["run_hash"],
                    "checkpoint_dir": model["checkpoint_dir"],
                    "selected_model_best_validation_score": model["best_validation_score"],
                    "selected_model_best_epoch": model["best_epoch"],
                    "seed": model["seed"],
                    "success": False,
                    "status": "skipped_missing_artifacts",
                    "message": message,
                    "missing_artifacts": missing_paths,
                    "prep_command": prep_commands[key],
                    "command": None,
                    "returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "metrics_file": "",
                    "metrics": {},
                    "loaded_epoch": None,
                }
                results.append(row)
                print(f"  -> SKIP: {message}")
                for mp in missing_paths:
                    print(f"     missing: {mp}")
                continue

            try:
                eval_out = run_inference_once(
                    model,
                    test_dataset=test_dataset,
                    epoch=args.epoch,
                    metrics_dir=metrics_dir,
                    timeout=args.inference_timeout_seconds,
                )
            except Exception as e:
                eval_out = {
                    "success": False,
                    "message": f"inference invocation failed: {e}",
                    "command": None,
                    "returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "metrics_file": "",
                    "metrics": {},
                    "loaded_epoch": None,
                }
            row = {
                "detection_system": system,
                "training_dataset": model["training_dataset"],
                "testing_dataset": test_dataset,
                "run_hash": model["run_hash"],
                "checkpoint_dir": model["checkpoint_dir"],
                "selected_model_best_validation_score": model["best_validation_score"],
                "selected_model_best_epoch": model["best_epoch"],
                "seed": model["seed"],
                "status": "evaluated" if eval_out["success"] else "failed",
                **eval_out,
            }
            results.append(row)

            status = "OK" if eval_out["success"] else "FAIL"
            print(f"  -> {status}: {eval_out['message']}")

    success_count = sum(1 for r in results if r["success"])
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "selected_models": selected,
            "dropped_duplicates": selection["dropped_duplicates"],
            "skipped_unknown": selection["skipped_unknown"],
        },
        "test_datasets": test_datasets,
        "results": results,
        "prep_commands": prep_commands,
        "summary": {
            "total_evaluations": len(results),
            "successful": success_count,
            "failed": len(results) - success_count,
            "skipped_missing_artifacts": sum(
                1 for r in results if r.get("status") == "skipped_missing_artifacts"
            ),
        },
    }

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print("\nSaved evaluation results to", results_path)
    print(
        f"Summary: successful={payload['summary']['successful']} "
        f"failed={payload['summary']['failed']} total={payload['summary']['total_evaluations']}"
    )


if __name__ == "__main__":
    main()
