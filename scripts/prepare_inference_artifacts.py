#!/usr/bin/env python3
"""Prepare prerequisite artifacts for standalone inference.

Runs only the prerequisite pipeline tasks:
construction -> transformation -> featurization -> feat_inference -> batching

This avoids executing training/evaluation/triage while ensuring inference inputs exist.
"""

import argparse

from pidsmaker.config import get_runtime_required_args, get_yml_cfg, set_task_to_done, update_task_paths_to_restart
from pidsmaker.tasks import batching, construction, feat_inference, featurization, transformation


TASKS = ["construction", "transformation", "featurization", "feat_inference", "batching"]
TASK_TO_MODULE = {
    "construction": construction,
    "transformation": transformation,
    "featurization": featurization,
    "feat_inference": feat_inference,
    "batching": batching,
}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare feat_inference and batching prerequisites for a model/dataset pair."
    )
    parser.add_argument("model", help="Detection system config name, e.g. velox")
    parser.add_argument("dataset", help="Dataset name, e.g. CADETS_E5 or optc_h201")
    parser.add_argument("--artifact-dir", default="./artifacts/")
    parser.add_argument("--database-host", default="localhost")
    parser.add_argument("--database-user", default="postgres")
    parser.add_argument("--database-password", default="postgres")
    parser.add_argument("--database-port", default="5432")
    parser.add_argument("--force-rebuild", action="store_true", help="Re-run prerequisite tasks even if done markers exist.")
    args = parser.parse_args()

    runtime_args = get_runtime_required_args(
        args=[
            args.model,
            args.dataset,
            "--artifact_dir",
            args.artifact_dir,
            "--database_host",
            args.database_host,
            "--database_user",
            args.database_user,
            "--database_password",
            args.database_password,
            "--database_port",
            args.database_port,
        ]
    )
    cfg = get_yml_cfg(runtime_args)

    should_restart = update_task_paths_to_restart(cfg)

    for task in TASKS:
        task_cfg = getattr(cfg, task)
        task_path = task_cfg._task_path
        must_run = args.force_rebuild or should_restart.get(task, True)

        if must_run:
            print(f"[RUN] {task} -> {task_path}")
            TASK_TO_MODULE[task].main(cfg)
            set_task_to_done(task_path)
        else:
            print(f"[SKIP] {task} already prepared -> {task_path}")

    print("\nPrepared paths:")
    print(f"  feat_inference edge embeds: {cfg.feat_inference._edge_embeds_dir}")
    print(f"  batching preprocessed graphs: {cfg.batching._preprocessed_graphs_dir}")


if __name__ == "__main__":
    main()
