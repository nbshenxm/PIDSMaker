"""Pack PIDSMaker per-batch edge-loss CSVs into one Parquet file per (split, epoch).

PIDSMaker writes one CSV per batch per epoch (e.g. ~2,700 files/epoch on
CADETS_E3), which exhausts the fileset's file-count quota after a few dozen
members. This replaces
    edge_losses/<split>/model_epoch_<k>/<t0>~<t1>.csv   (thousands of files)
with
    edge_losses/<split>/model_epoch_<k>.parquet         (one file)
keeping the batch order in column `b` and the original file names in the
Parquet metadata. Row counts are verified before the CSV directory is removed.

  python -m caliber.pack --artifact_dir DIR            # every member with a manifest
  python -m caliber.pack --task_path TRAINING_TASK_DIR # one member
"""

import argparse
import glob
import json
import os
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EDGE_COLS = ["loss", "srcnode", "dstnode", "time", "edge_type"]


def pack_epoch_dir(d, out=None):
    """Pack CSV directory d into Parquet file out (default: d + '.parquet'), then remove d."""
    files = sorted(f for f in os.listdir(d) if f.endswith(".csv"))
    if not files:
        return 0
    dfs = []
    for b, fn in enumerate(files):
        df = pd.read_csv(os.path.join(d, fn))
        df = df[[c for c in EDGE_COLS if c in df.columns]]
        df["b"] = b
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"caliber_batch_files"] = json.dumps(files).encode()
    table = table.replace_schema_metadata(meta)
    out = out or (d.rstrip("/") + ".parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    pq.write_table(table, tmp, compression="zstd")
    if pq.ParquetFile(tmp).metadata.num_rows != len(df):
        os.remove(tmp)
        raise RuntimeError(f"row count mismatch packing {d}")
    os.replace(tmp, out)
    shutil.rmtree(d)
    return len(files)


def pack_task(task_path, log=print, src_edge_dir=None):
    """src_edge_dir: where the CSVs are (node-local scratch, see CALIBER_EDGE_SCRATCH).
    Parquet files always go to <task_path>/edge_losses/<split>/model_epoch_<k>.parquet."""
    src = src_edge_dir or os.path.join(task_path, "edge_losses")
    dst = os.path.join(task_path, "edge_losses")
    removed = 0
    for split in ("val", "test"):
        for d in sorted(glob.glob(os.path.join(src, split, "model_epoch_*"))):
            if os.path.isdir(d):
                out = os.path.join(dst, split, os.path.basename(d.rstrip("/")) + ".parquet")
                removed += pack_epoch_dir(d, out)
    log(f"packed {task_path}: {removed} CSV files -> parquet")
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir")
    ap.add_argument("--task_path", action="append", default=[])
    args = ap.parse_args()
    paths = list(args.task_path)
    if args.artifact_dir:
        for p in glob.glob(os.path.join(args.artifact_dir, "caliber", "manifests", "*.json")):
            with open(p, encoding="utf-8") as f:
                paths.append(json.load(f)["training_task_path"])
    total = sum(pack_task(p) for p in sorted(set(paths)))
    print(f"total CSV files removed: {total}")


if __name__ == "__main__":
    main()
