"""Load PIDSMaker outputs (no PIDSMaker import).

Each ensemble member leaves per-epoch edge-loss CSVs:
    <training_task_path>/edge_losses/{val,test}/model_epoch_<k>/<t0>~<t1>.csv
with columns loss, srcnode, dstnode, time, edge_type. One CSV = one PIDSMaker
batch ("time window"); files sort chronologically by name.
"""

import glob
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

EDGE_COLS = ["loss", "srcnode", "dstnode", "time", "edge_type"]


# ----------------------------------------------------------------------------- manifests
def load_manifests(artifact_dir, system=None, dataset=None, group=None):
    out = []
    for p in sorted(glob.glob(os.path.join(artifact_dir, "caliber", "manifests", "*.json"))):
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        if system and m["system"] != system:
            continue
        if dataset and m["dataset"] != dataset:
            continue
        if group and m["group"] != group:
            continue
        out.append(m)
    return out


def group_manifests(manifests):
    """(system, dataset, group) -> members sorted by seed."""
    groups = defaultdict(list)
    for m in manifests:
        groups[(m["system"], m["dataset"], m["group"])].append(m)
    return {k: sorted(v, key=lambda m: m["seed"]) for k, v in groups.items()}


def load_labels(artifact_dir, dataset, gt_version):
    d = os.path.join(artifact_dir, "caliber", "labels", f"{dataset}__{gt_version}")
    with open(os.path.join(d, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    meta_path = os.path.join(d, "node_meta.csv.gz")
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    return labels, meta


# ----------------------------------------------------------------------------- edge losses
def list_epochs(edge_losses_dir, split):
    d = os.path.join(edge_losses_dir, split)
    if not os.path.isdir(d):
        return []
    return sorted(int(x.rsplit("_", 1)[-1]) for x in os.listdir(d) if x.startswith("model_epoch_"))


def _cache_path(cache_dir, edge_losses_dir, split, epoch):
    key = hashlib.sha1(f"{os.path.abspath(edge_losses_dir)}|{split}|{epoch}".encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"{key}.parquet")


def read_split(edge_losses_dir, split, epoch, cache_dir=None):
    """All edges of one split/epoch as a DataFrame; column `b` = batch index (file order)."""
    if cache_dir:
        cp = _cache_path(cache_dir, edge_losses_dir, split, epoch)
        if os.path.exists(cp):
            return pd.read_parquet(cp)
    d = os.path.join(edge_losses_dir, split, f"model_epoch_{epoch}")
    files = sorted(os.listdir(d))
    dfs = []
    for b, fn in enumerate(files):
        df = pd.read_csv(os.path.join(d, fn))
        df = df[[c for c in EDGE_COLS if c in df.columns]]
        df["b"] = b
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["loss"] = df["loss"].astype(np.float32)
    df["b"] = df["b"].astype(np.int32)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(cp, index=False)
    return df


def mean_loss(edge_losses_dir, split, epoch, cache_dir=None):
    return float(read_split(edge_losses_dir, split, epoch, cache_dir)["loss"].mean())


def select_epoch(manifest, rule="min_val_loss", cache_dir=None):
    """Pick the member's epoch WITHOUT test labels.

    min_val_loss: epoch with the lowest mean benign validation edge loss.
    last:         final epoch.
    """
    epochs = sorted(set(manifest["val_epochs"]) & set(manifest["test_epochs"]))
    if not epochs:
        epochs = sorted(set(list_epochs(manifest["edge_losses_dir"], "val"))
                        & set(list_epochs(manifest["edge_losses_dir"], "test")))
    if not epochs:
        raise FileNotFoundError(f"no epochs with both val and test losses in {manifest['edge_losses_dir']}")
    if rule == "last":
        return epochs[-1]
    if rule == "min_val_loss":
        losses = {e: mean_loss(manifest["edge_losses_dir"], "val", e, cache_dir) for e in epochs}
        return min(losses, key=losses.get)
    raise ValueError(rule)


# ----------------------------------------------------------------------------- aggregation
def _node_long(df, use_dst=True):
    """One row per (edge endpoint): node, loss, b, time, edge_type."""
    cols = ["loss", "b", "time", "edge_type"]
    src = df[cols].assign(node=df["srcnode"].values)
    if not use_dst:
        return src
    dst = df[cols].assign(node=df["dstnode"].values)
    return pd.concat([src, dst], ignore_index=True)


def window_ids(df, window="batch"):
    """Window assignment: 'batch' (PIDSMaker file) or '<N>min' fixed wall-clock bins."""
    if window == "batch":
        return df["b"].values
    if window.endswith("min"):
        width = int(window[:-3]) * 60 * 10**9
        return (df["time"].values // width).astype(np.int64)
    raise ValueError(window)


def node_scores(df, agg="max", use_dst=True):
    """PIDSMaker node-level score over the whole split (default: max incident edge loss)."""
    long = _node_long(df, use_dst)
    g = long.groupby("node")["loss"]
    out = pd.DataFrame({"score": getattr(g, agg)(), "n_edges": g.size()})
    return out


def node_window_scores(df, window="batch", agg="max", use_dst=True):
    long = _node_long(df, use_dst)
    long["w"] = window_ids(long, window)
    g = long.groupby(["w", "node"])["loss"]
    return pd.DataFrame({"score": getattr(g, agg)(), "n_edges": g.size()}).reset_index()


def window_scores(nw, topk=5):
    """Window score = mean of its top-k node scores (insensitive to window size)."""
    def topk_mean(s):
        v = np.sort(s.values)[::-1][:topk]
        return v.mean()
    g = nw.groupby("w")["score"]
    return pd.DataFrame({"score": g.apply(topk_mean), "n_nodes": g.size()})
