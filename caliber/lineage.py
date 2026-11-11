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


def load_owners(artifact_dir, dataset):
    """file/socket -> the process that interacts with it most (node-level detectors report no
    edges, so a non-process unit has no partner of its own)."""
    p = os.path.join(artifact_dir, "caliber", "labels", f"{dataset}__owners.csv.gz")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return pd.Series(df["process"].astype(np.int64).values, index=df["object"].astype(np.int64).values)


def case_keys(st, artifact_dir, dataset, meta, k=1):
    """Case key per unit: k-th ancestor of the acting process (Section 'Cases, not units').
    Returns (keys, has_lineage). Without a process tree, cases fall back to entities."""
    node = st["node"].astype(np.int64)
    parents = load_parents(artifact_dir, dataset)
    if parents is None:
        return node, False
    partner = st.get("partner")
    partner = node.copy() if partner is None else np.asarray(partner, dtype=np.int64)
    selfloop = partner == node
    if selfloop.any():
        owners = load_owners(artifact_dir, dataset)
        if owners is not None:
            mapped = pd.Series(node[selfloop]).map(owners)
            partner = partner.copy()
            partner[selfloop] = mapped.fillna(pd.Series(node[selfloop])).astype(np.int64).values
    return lineage_keys(node, partner, parents, process_predicate(parents, meta), k), True


def _acting_process(node, partner, is_process, owners):
    """The process behind each unit: the unit itself if it is a process, else its partner process
    (other endpoint of its most anomalous edge), else, for detectors without edges, its owner."""
    node = np.asarray(node, dtype=np.int64)
    partner = node.copy() if partner is None else np.asarray(partner, dtype=np.int64)
    if owners is not None:
        selfloop = partner == node
        if selfloop.any():
            mapped = pd.Series(node[selfloop]).map(owners)
            partner = partner.copy()
            partner[selfloop] = mapped.fillna(pd.Series(node[selfloop])).astype(np.int64).values
    return np.where(is_process(node), node, np.where(is_process(partner), partner, -1))


def executable_keys(st, artifact_dir, dataset, meta):
    """Executable (process image path) behind each test and validation unit, as integer keys;
    units without a known process get a unique negative key. Used by the act tier's novelty gate."""
    if meta is None or "type" not in meta:
        return -(st["node"].astype(np.int64) + 1), -(st["node_val"].astype(np.int64) + 1)
    sub = meta[meta["type"] == "subject"]
    path_of = pd.Series(sub["path"].fillna("").astype(str).values, index=sub["node"].astype(np.int64).values)
    procs = set(path_of.index)
    is_proc = lambda x: np.isin(np.asarray(x, dtype=np.int64), np.fromiter(procs, dtype=np.int64))
    owners = load_owners(artifact_dir, dataset)
    codes = {}
    def keys(node, partner):
        p = _acting_process(node, partner, is_proc, owners)
        paths = pd.Series(p).map(path_of).fillna("")
        out = np.empty(len(p), dtype=np.int64)
        for i, (pp, path) in enumerate(zip(p, paths)):
            if pp < 0 or not path:
                out[i] = -(int(node[i]) + 1)
            else:
                out[i] = codes.setdefault(path, len(codes))
        return out
    return keys(st["node"], st.get("partner")), keys(st["node_val"], None)
