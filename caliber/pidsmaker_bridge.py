"""The only CALIBER module that imports PIDSMaker.

Subcommands (pass the same SYSTEM DATASET [--x.y=z ...] args as pidsmaker/main.py):

  manifest  Record where one ensemble member's outputs live:
            {artifact_dir}/caliber/manifests/<system>__<dataset>__<group>__seed<k>.json
  labels    Export ground truth for a dataset (needs the DB):
            {artifact_dir}/caliber/labels/<dataset>__<gt_version>/labels.json
            {artifact_dir}/caliber/labels/<dataset>__<gt_version>/node_meta.csv.gz

`group` hashes every CLI arg except the seed and the machine-specific ones
(artifact dir, DB host/port), so members of one ensemble share a group while
e.g. a `--training.stable_optim=True` ensemble gets its own.
"""

import argparse
import gzip
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

MACHINE_ARGS = {"--artifact_dir", "--database_host", "--database_port",
                "--database_user", "--database_password"}
SEED_ARGS = {"--training.seed"}


def _strip_args(argv, drop):
    """Remove `--k v` and `--k=v` occurrences of the given keys."""
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        key = a.split("=", 1)[0]
        if key in drop:
            if "=" not in a:
                skip = True
            continue
        out.append(a)
    return out


def group_id(argv):
    rest = _strip_args(argv[2:], MACHINE_ARGS | SEED_ARGS)
    canon = " ".join(sorted(rest))
    return hashlib.sha1(canon.encode()).hexdigest()[:10], canon


def load_cfg(argv):
    from pidsmaker.config import get_runtime_required_args, get_yml_cfg

    args = get_runtime_required_args(args=argv)
    return get_yml_cfg(args)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _list_epochs(edge_losses_dir, split):
    d = os.path.join(edge_losses_dir, split)
    if not os.path.isdir(d):
        return []
    return sorted({int(x.split(".")[0].rsplit("_", 1)[-1]) for x in os.listdir(d)
                   if x.startswith("model_epoch_") and not x.endswith(".tmp")})


def cmd_manifest(argv):
    cfg = load_cfg(argv)
    system, dataset = argv[0], argv[1]
    gid, canon = group_id(argv)
    edge_dir = cfg.training._edge_losses_dir
    m = {
        "system": system,
        "dataset": dataset,
        "group": gid,
        "group_args": canon,
        "seed": int(cfg.training.seed),
        "argv": argv,
        "training_task_path": cfg.training._task_path,
        "edge_losses_dir": edge_dir,
        "evaluation_task_path": cfg.evaluation._task_path,
        "val_epochs": _list_epochs(edge_dir, "val"),
        "test_epochs": _list_epochs(edge_dir, "test"),
        "ground_truth_version": str(cfg.evaluation.ground_truth_version),
        "threshold_method": str(cfg.evaluation.node_evaluation.threshold_method),
        "train_dates": list(cfg.dataset.train_dates),
        "val_dates": list(cfg.dataset.val_dates),
        "test_dates": list(cfg.dataset.test_dates),
        "git": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = os.path.join(_artifact_dir(argv), "caliber", "manifests")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{system}__{dataset}__{gid}__seed{m['seed']}.json")
    with open(out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"manifest: {out}  (epochs val={len(m['val_epochs'])} test={len(m['test_epochs'])})")
    if os.environ.get("CALIBER_PACK") == "1":
        # Collapse per-batch CSVs into one Parquet per (split, epoch): file-count quota.
        from caliber.pack import pack_task
        pack_task(cfg.training._task_path)


def _artifact_dir(argv):
    for i, a in enumerate(argv):
        if a == "--artifact_dir":
            return argv[i + 1]
        if a.startswith("--artifact_dir="):
            return a.split("=", 1)[1]
    raise ValueError("--artifact_dir is required")


def cmd_labels(argv):
    from pidsmaker.utils.labelling import get_GP_of_each_attack, ground_truth_attacks
    from pidsmaker.utils.dataset_utils import get_node_to_path_and_type

    cfg = load_cfg(argv)
    dataset = argv[1]
    gt_version = str(cfg.evaluation.ground_truth_version)
    out_dir = os.path.join(_artifact_dir(argv), "caliber", "labels", f"{dataset}__{gt_version}")
    os.makedirs(out_dir, exist_ok=True)

    attacks = get_GP_of_each_attack(cfg)
    files = ground_truth_attacks(cfg)
    payload = {
        "dataset": dataset,
        "ground_truth_version": gt_version,
        "timezone": str(cfg.dataset.timezone),
        "train_dates": list(cfg.dataset.train_dates),
        "val_dates": list(cfg.dataset.val_dates),
        "test_dates": list(cfg.dataset.test_dates),
        "attacks": [
            {
                "index": int(i),
                "file": files[i][0],
                "start_ns": int(a["time_range"][0]),
                "end_ns": int(a["time_range"][1]),
                "nids": sorted(int(n) for n in a["nids"]),
            }
            for i, a in attacks.items()
        ],
    }
    payload["malicious_nids"] = sorted({n for a in payload["attacks"] for n in a["nids"]})
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(payload, f)
    print(f"labels: {len(payload['malicious_nids'])} malicious nodes over {len(payload['attacks'])} attacks")

    meta = get_node_to_path_and_type(cfg)
    with gzip.open(os.path.join(out_dir, "node_meta.csv.gz"), "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "type", "path"])
        for nid, info in meta.items():
            w.writerow([int(nid), info.get("type", ""), info.get("path", "")])
    print(f"node_meta: {len(meta)} nodes -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["manifest", "labels"])
    ns, rest = p.parse_known_args()
    if len(rest) < 2:
        sys.exit("usage: python -m caliber.pidsmaker_bridge {manifest,labels} SYSTEM DATASET [args...]")
    {"manifest": cmd_manifest, "labels": cmd_labels}[ns.command](rest)


if __name__ == "__main__":
    main()
