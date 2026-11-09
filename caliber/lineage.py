"""Provenance lineage keys for incident aggregation.

Score-level signals cannot tell an alert flood from a genuine upward shift of the score tail.
Provenance structure can: activity an attacker generates from one foothold descends from the
same process, while benign drift spans unrelated lineages. A lineage key maps every alert unit
to an ancestor process; incidents are then counted per (lineage, hour) instead of per
(entity, hour), so a flood from one foothold opens one incident however many processes it
spawns.

Key of a unit (entity n, partner p = the other endpoint of n's highest-loss edge):
  process n            -> k-th ancestor of n (k=0: n itself)
  non-process n        -> k-th ancestor of p if p is a process, else n
"""

import gzip
import os

import numpy as np
import pandas as pd


def load_parents(artifact_dir, dataset):
    p = os.path.join(artifact_dir, "caliber", "labels", f"{dataset}__lineage.csv.gz")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    return dict(zip(df["child"].astype(np.int64), df["parent"].astype(np.int64)))


def ancestor(n, parents, k):
    for _ in range(k):
        p = parents.get(n)
        if p is None or p == n:
            break
        n = p
    return n


def lineage_keys(node, partner, parents, is_process, k=1):
    """Vectorised over units; is_process(x) -> bool array."""
    node = np.asarray(node, dtype=np.int64)
    partner = np.asarray(partner, dtype=np.int64)
    proc_n = is_process(node)
    proc_p = is_process(partner)
    base = np.where(proc_n, node, np.where(proc_p, partner, node))
    cache = {}
    out = np.empty(len(base), dtype=np.int64)
    for i, b in enumerate(base.tolist()):
        v = cache.get(b)
        if v is None:
            v = ancestor(b, parents, k)
            cache[b] = v
        out[i] = v
    return out


def process_predicate(parents, node_meta=None):
    """Process test: node type 'subject' in node_meta if available, else appears in the tree."""
    if node_meta is not None and "type" in node_meta:
        procs = set(node_meta.loc[node_meta["type"] == "subject", "node"].astype(np.int64))
    else:
        procs = set(parents.keys()) | set(parents.values())
    procs_arr = np.fromiter(procs, dtype=np.int64)
    return lambda x: np.isin(np.asarray(x, dtype=np.int64), procs_arr)
