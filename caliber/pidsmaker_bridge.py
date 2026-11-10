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
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

MACHINE_ARGS = {"--artifact_dir", "--database_host", "--database_port", "--weights_dir",
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
    edge_dir = cfg.training._edge_losses_dir            # where the CSVs are now (maybe scratch)
    final_dir = getattr(cfg.training, "_edge_losses_final_dir", edge_dir)
    m = {
        "system": system,
        "dataset": dataset,
        "group": gid,
        "group_args": canon,
        "seed": int(cfg.training.seed),
        "argv": argv,
        "training_task_path": cfg.training._task_path,
        "edge_losses_dir": final_dir,
        "evaluation_task_path": cfg.evaluation._task_path,
        "val_epochs": _list_epochs(edge_dir, "val"),
        "test_epochs": _list_epochs(edge_dir, "test"),
        "ground_truth_version": str(cfg.evaluation.ground_truth_version),
        "mimicry_edge_num": int(cfg.construction.mimicry_edge_num or 0),
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
        pack_task(cfg.training._task_path, src_edge_dir=edge_dir, tz=str(cfg.dataset.timezone))
    elif edge_dir != final_dir:
        raise RuntimeError("edge losses are on node-local scratch; set CALIBER_PACK=1 to keep them")


def cmd_drop_unpacked(argv):
    """After a failed run whose edge losses lived on node-local scratch: if training finished
    (done.txt) but no packed test losses exist, delete the training task so a retry retrains
    instead of evaluating losses that are gone."""
    import shutil
    cfg = load_cfg(argv)
    tp = cfg.training._task_path
    final_dir = getattr(cfg.training, "_edge_losses_final_dir", cfg.training._edge_losses_dir)
    if cfg.training._edge_losses_dir == final_dir or not os.path.exists(os.path.join(tp, "done.txt")):
        return
    if not glob.glob(os.path.join(final_dir, "test", "model_epoch_*.parquet")):
        shutil.rmtree(tp)
        print(f"dropped unpacked training task {tp}")


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
    from caliber.data import labels_key
    out_dir = os.path.join(_artifact_dir(argv), "caliber", "labels",
                           labels_key(dataset, gt_version, cfg.construction.mimicry_edge_num or 0))
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


# Process-creation operations per data source (DARPA TC, EDR datasets, OpTC). Links are kept
# only when both endpoints are processes (OpTC's CREATE also covers files).
PROCESS_CREATION_OPS = ("EVENT_FORK", "EVENT_CLONE", "PROCESS_EXECUTED", "CREATE")


def cmd_lineage(argv):
    """Export the process tree (child -> parent, first creation event) from the raw events.

    PIDSMaker's graphs drop process-creation events (they are not in its edge vocabulary), so
    the lineage must come from the database. Written to
    {artifact_dir}/caliber/labels/<dataset>__lineage.csv.gz with columns child,parent,time."""
    from pidsmaker.utils.utils import init_database_connection

    cfg = load_cfg(argv)
    dataset = argv[1]
    cur, conn = init_database_connection(cfg)
    cur.execute("SELECT index_id FROM subject_node_table")
    procs = {str(r[0]) for r in cur.fetchall()}
    cur.execute(
        "SELECT dst_index_id, src_index_id, MIN(timestamp_rec) FROM event_table "
        "WHERE operation IN %s AND src_index_id <> dst_index_id "
        "GROUP BY dst_index_id, src_index_id", (PROCESS_CREATION_OPS,))
    rows = [r for r in cur.fetchall() if str(r[0]) in procs and str(r[1]) in procs]
    # keep the earliest creation event per child
    first = {}
    for child, parent, t in rows:
        child, parent, t = int(child), int(parent), int(t)
        if child not in first or t < first[child][1]:
            first[child] = (parent, t)
    out_dir = os.path.join(_artifact_dir(argv), "caliber", "labels")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{dataset}__lineage.csv.gz")
    with gzip.open(out, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["child", "parent", "time"])
        for child, (parent, t) in first.items():
            w.writerow([child, parent, t])
    print(f"lineage: {len(first)} child->parent links ({len(rows)} creation pairs) -> {out}")


def cmd_train_procs(argv):
    """Executables seen during the TRAINING period (allowlist baseline, Guerra et al.).

    Writes {artifact_dir}/caliber/labels/<dataset>__train_procs.csv.gz with one row per process
    active in a training day: node, path (executable path from subject_node_table)."""
    from datetime import datetime, timedelta
    from pidsmaker.utils.utils import datetime_to_ns_time_US, init_database_connection

    cfg = load_cfg(argv)
    dataset = argv[1]
    tz = str(getattr(cfg.dataset, "timezone", "") or "")
    if tz.upper() == "UTC":
        import pytz
        to_ns = lambda s: int(pytz.utc.localize(datetime.strptime(s, "%Y-%m-%d %H:%M:%S")).timestamp() * 1e9)
    else:
        to_ns = datetime_to_ns_time_US
    cur, conn = init_database_connection(cfg)
    cur.execute("SELECT index_id, path FROM subject_node_table")
    path_of = {str(r[0]): str(r[1]) for r in cur.fetchall()}
    active = set()
    for d in cfg.dataset.train_dates:
        nxt = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        cur.execute("SELECT DISTINCT src_index_id FROM event_table "
                    "WHERE timestamp_rec >= %s AND timestamp_rec < %s",
                    (to_ns(f"{d} 00:00:00"), to_ns(f"{nxt} 00:00:00")))
        active |= {str(r[0]) for r in cur.fetchall() if str(r[0]) in path_of}
    out_dir = os.path.join(_artifact_dir(argv), "caliber", "labels")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{dataset}__train_procs.csv.gz")
    with gzip.open(out, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "path"])
        for n in sorted(active, key=int):
            w.writerow([n, path_of[n]])
    print(f"train_procs: {len(active)} processes active on {len(cfg.dataset.train_dates)} training days -> {out}")


def main():
    p = argparse.ArgumentParser()
    cmds = {"manifest": cmd_manifest, "labels": cmd_labels, "lineage": cmd_lineage,
            "drop_unpacked": cmd_drop_unpacked, "train_procs": cmd_train_procs}
    p.add_argument("command", choices=list(cmds))
    ns, rest = p.parse_known_args()
    if len(rest) < 2:
        sys.exit("usage: python -m caliber.pidsmaker_bridge {" + ",".join(cmds) + "} SYSTEM DATASET [args...]")
    cmds[ns.command](rest)


if __name__ == "__main__":
    main()
