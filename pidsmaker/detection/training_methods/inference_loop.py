import os
import random
import time
import tracemalloc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from pidsmaker.utils.utils import (
    calculate_average_from_file,
    get_device,
    log,
    log_tqdm,
    ns_time_to_datetime_US,
    set_seed,
)


class OnlineUpdater:
    """CALIBER: gated online learning during test inference (test-then-train).

    Each test window is first scored (and its losses written) by the current model; only then
    does the model take `steps` optimizer steps on the window's edges that are confidently benign,
    i.e. whose loss is below the `gate` quantile of this model's own benign validation losses.
    Edges above the gate (possible attacks, or simply unfamiliar) are never trained on, so the
    detector adapts to benign drift without learning what it is unsure about.
    spec (JSON): {"lr": 1e-5, "gate": 0.9, "steps": 1, "min_edges": 32}; lr = 0 only records."""

    def __init__(self, cfg, model, val_losses, device):
        import json
        if not cfg._from_weights:
            # updating during training-time evaluation would leak into later training epochs
            raise ValueError("--online requires --from_weights (frozen members scored test-then-train)")
        spec = json.loads(cfg._online)
        self.lr = float(spec.get("lr", 1e-5))
        self.steps = int(spec.get("steps", 1))
        self.min_edges = int(spec.get("min_edges", 32))
        # anchor: L2 pull toward the trained weights (L2-SP), the standard guard against drift and
        # forgetting in continual / test-time adaptation; every: update on every n-th window only
        self.anchor = float(spec.get("anchor", 0.0))
        self.every = max(1, int(spec.get("every", 1)))
        self.n_seen = 0
        self.theta0 = [p.detach().clone() for p in model.parameters()] if self.anchor > 0 else None
        q = float(spec.get("gate", 0.9))
        self.tau = float(np.quantile(np.asarray(val_losses, float), q)) if len(val_losses) else np.inf
        self.model = model
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=self.lr) if self.lr > 0 else None
        self.n_upd = self.n_edges = self.n_held = 0
        enc = getattr(model, "encoder", None)
        if self.opt is not None and enc is not None and "TGN" in type(enc).__name__:
            raise NotImplementedError("online updates on TGN encoders need a memory snapshot")
        log(f"[online] lr={self.lr} gate q={q} -> loss < {self.tau:.4f}, steps={self.steps}")

    def update(self, g, losses):
        if self.opt is None:
            return
        self.n_seen += 1
        if self.n_seen % self.every:
            return
        keep = np.asarray(losses, float) < self.tau
        self.n_held += int((~keep).sum())
        if keep.sum() < self.min_edges:
            return
        # Per-edge losses on the whole window (slicing a reindexed batch by edges also slices its
        # node tensors), with gradients on; average only over the confidently benign edges.
        mask = torch.from_numpy(keep).to(g.edge_index.device)
        self.model.train()
        self.model._force_grad = True
        try:
            for _ in range(self.steps):
                self.opt.zero_grad()
                per_edge = self.model(g, inference=True)["loss"]
                loss = per_edge[mask].mean()
                if self.theta0 is not None:
                    loss = loss + self.anchor * sum(((p - p0) ** 2).sum()
                                                    for p, p0 in zip(self.model.parameters(), self.theta0))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
        finally:
            self.model._force_grad = False
            self.model.eval()
        self.n_upd += 1
        self.n_edges += int(keep.sum())

    def summary(self):
        log(f"[online] {self.n_upd} updates on {self.n_edges} benign-gated edges; "
            f"{self.n_held} edges held out (above the gate)")


@torch.no_grad()
def test_edge_level(
    data,
    model,
    split,
    model_epoch_file,
    cfg,
    device,
):
    model.eval()

    start_time = data.t[0]
    all_losses = []

    validation = split == "val"

    results = model(data, inference=True, validation=validation)
    each_edge_loss = results["loss"]
    all_losses.extend(each_edge_loss.cpu().numpy().tolist())

    # If the data has been reindexed in the loader or batched, we retrieve original node IDs
    # to later find the labels
    edge_index = data.original_edge_index

    srcnodes = edge_index[0, :].cpu().numpy()
    dstnodes = edge_index[1, :].cpu().numpy()
    t_vars = data.t.cpu().numpy()
    losses = each_edge_loss.cpu().numpy()
    edge_types = (data.edge_type.max(dim=1).indices + 1).cpu().numpy()

    # if 1 in data.y:
    #     log(f"Mean score of fake malicious edges: {losses[data.y.cpu() == 1].mean():.4f}")
    #     log(f"Mean score of benign malicious edges: {losses[data.y.cpu() == 0].mean():.4f}")

    edge_df = pd.DataFrame(
        {
            "loss": losses.astype(float),
            "srcnode": srcnodes.astype(int),
            "dstnode": dstnodes.astype(int),
            "time": t_vars.astype(int),
            "edge_type": edge_types.astype(int),
        }
    )
    if "entropy" in results and results["entropy"].numel() == len(edge_df):
        edge_df["entropy"] = results["entropy"].cpu().numpy().astype(np.float32)   # CALIBER
    if "abstain" in results and results["abstain"].numel() == len(edge_df):
        edge_df["abstain"] = results["abstain"].cpu().numpy().astype(np.float32)   # CALIBER
    for c in ("novelty", "novelty_y"):                                             # CALIBER
        if c in results and results[c].numel() == len(edge_df):
            edge_df[c] = results[c].cpu().numpy().astype(np.float32)

    # Here is a checkpoint, which records all edge losses in the current time window
    time_interval = (
        ns_time_to_datetime_US(start_time, timezone=cfg.dataset.timezone) + "~" + ns_time_to_datetime_US(edge_df["time"].max(), timezone=cfg.dataset.timezone)
    )

    logs_dir = os.path.join(cfg.training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    csv_file = os.path.join(logs_dir, time_interval + ".csv")

    edge_df.to_csv(csv_file, sep=",", header=True, index=False, encoding="utf-8")
    return all_losses


@torch.no_grad()
def test_node_level(
    data,
    model,
    split,
    model_epoch_file,
    cfg,
    device,
):
    model.eval()

    node_list = []
    start_time = data.t[0]
    end_time = data.t[-1]
    losses = []

    validation = split == "val"

    data = data.to(device)

    results = model(data, inference=True, validation=validation)
    loss = results["loss"]
    losses.extend(loss.cpu().numpy().tolist())
    n_id = getattr(data, "original_n_id_tgn", getattr(data, "original_n_id"))

    # ThreaTrace code
    if cfg.evaluation.node_evaluation.threshold_method == "threatrace":
        out = results["out"]
        pred = out.max(1)[1]
        pro = F.softmax(out, dim=1)
        pro1 = pro.max(1)
        for i in range(len(out)):
            pro[i][pro1[1][i]] = -1
        pro2 = pro.max(1)

        node_type_num = data.node_type.argmax(1)
        for i in range(len(out)):
            if pro2[0][i] != 0:
                score = pro1[0][i] / pro2[0][i]
            else:
                score = pro1[0][i] / 1e-5
            score = torch.log(score + 1e-12)  # we do that or the score is much too high
            score = max(score.item(), 0)

            node = n_id[i].item()
            correct_pred = int((node_type_num[i] == pred[i]).item())

            temp_dic = {
                "node": node,
                "loss": float(loss[i].item()),
                "threatrace_score": score,
                "correct_pred": correct_pred,
            }
            node_list.append(temp_dic)

    # Flash code
    elif cfg.evaluation.node_evaluation.threshold_method == "flash":
        out = results["out"]
        pred = out.max(1)[1]
        sorted, indices = out.sort(dim=1, descending=True)
        eps = 1e-6
        conf = (sorted[:, 0] - sorted[:, 1]) / (sorted[:, 0] + eps)
        conf = (conf - conf.min()) / conf.max() if conf.max() > 0 else conf

        node_type_num = data.node_type.argmax(1)
        for i in range(len(out)):
            score = max(conf[i].item(), 0)

            node = n_id[i].item()
            correct_pred = int((node_type_num[i] == pred[i]).item())

            temp_dic = {
                "node": node,
                "loss": float(loss[i].item()),
                "flash_score": score,
                "correct_pred": correct_pred,
            }
            node_list.append(temp_dic)

    # Magic codes
    elif cfg.evaluation.node_evaluation.threshold_method == "magic":
        os.makedirs(cfg.training._magic_dir, exist_ok=True)
        if split == "val":
            x_train, *_ = model.embed(data, inference=True)
            x_train = x_train.cpu().numpy()
            num_nodes = x_train.shape[0]
            sample_size = 5000 if num_nodes > 5000 else num_nodes
            sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
            x_train_sampled = x_train[sample_indices]
            x_train_mean = x_train_sampled.mean(axis=0)
            x_train_std = x_train_sampled.std(axis=0)
            x_train_sampled = (x_train_sampled - x_train_mean) / x_train_std

            x_train_sampled = pd.DataFrame.from_records(x_train_sampled)

            n_neighbors = 10
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train_sampled)
            idx = list(range(x_train_sampled.shape[0]))
            random.shuffle(idx)
            try:
                sample = x_train_sampled.iloc[idx[: min(50000, x_train_sampled.shape[0])]]
                distances_train, _ = nbrs.kneighbors(
                    sample, n_neighbors=min(len(sample), n_neighbors)
                )
            except KeyError as e:
                log(f"KeyError encountered: {e}")
                log(f"Available columns in x_train: {x_train_sampled.columns}")
                raise
            mean_distance_train = distances_train.mean().mean()
            if mean_distance_train == 0:
                mean_distance_train = 1e-9

            train_distance_file = os.path.join(cfg.training._magic_dir, "train_distance.txt")
            with open(train_distance_file, "a") as f:
                f.write(f"{mean_distance_train}\n")

            # CALIBER: also score validation nodes exactly as test nodes are scored (kNN distance
            # to a standardized sample of the same window, relative to the benign mean distance),
            # so that thresholds can be calibrated on benign validation scores.
            # (tiny windows can hold fewer nodes than n_neighbors, as the training-distance
            # query above already allows for)
            val_dist, _ = nbrs.kneighbors(
                (x_train - x_train_mean) / x_train_std,
                n_neighbors=min(n_neighbors, x_train_sampled.shape[0]),
            )
            val_score = val_dist.mean(axis=1) / mean_distance_train

            for i, node in enumerate(n_id):
                temp_dic = {
                    "node": node.item(),
                    "magic_score": float(val_score[i]),
                    "loss": float(loss[i].item()),
                }
                node_list.append(temp_dic)

        elif split == "test":
            train_distance_file = os.path.join(cfg.training._magic_dir, "train_distance.txt")
            mean_distance_train = calculate_average_from_file(train_distance_file)

            x_test, *_ = model.embed(data, inference=True)
            x_test = x_test.cpu().numpy()
            num_nodes = x_test.shape[0]
            sample_size = 5000 if num_nodes > 5000 else num_nodes
            sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
            x_test_sampled = x_test[sample_indices]
            x_test_mean = x_test_sampled.mean(axis=0)
            x_test_std = x_test_sampled.std(axis=0)
            x_test_sampled = (x_test_sampled - x_test_mean) / x_test_std

            torch.cuda.empty_cache()
            x_test_sampled = pd.DataFrame.from_records(x_test_sampled)

            n_neighbors = 10
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_test_sampled)

            # Query in the same standardized space as the fitted sample.
            distances, _ = nbrs.kneighbors(
                (x_test - x_test_mean) / x_test_std, n_neighbors=n_neighbors
            )
            distances = distances.mean(axis=1)
            # distances = distances.to_numpy()
            score = distances / mean_distance_train
            score = score.tolist()

            for i, node in enumerate(n_id):
                temp_dic = {
                    "node": node.item(),
                    "magic_score": float(score[i]),
                    "loss": float(loss[i].item()),
                }
                node_list.append(temp_dic)

    else:
        for i, node in enumerate(n_id):
            temp_dic = {
                "node": node.item(),
                "loss": float(loss[i].item()),
            }
            node_list.append(temp_dic)

    time_interval = ns_time_to_datetime_US(start_time, timezone=cfg.dataset.timezone) + "~" + ns_time_to_datetime_US(end_time, timezone=cfg.dataset.timezone)

    logs_dir = os.path.join(cfg.training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    csv_file = os.path.join(logs_dir, time_interval + ".csv")

    df = pd.DataFrame(node_list)
    df.to_csv(csv_file, sep=",", header=True, index=False, encoding="utf-8")
    return losses


def main(cfg, model, val_data, test_data, epoch, split, logging=True, train_data=None):
    set_seed(cfg)

    if split == "all":
        splits = [(val_data, "val"), (test_data, "test")]
    elif split == "val":
        splits = [(val_data, "val")]
    elif split == "test":
        splits = [(test_data, "test")]
    elif split == "train":
        splits = [(train_data, "train")]
    else:
        raise ValueError(f"Invalid split {split}")

    inference_device = cfg.training.inference_device
    if inference_device is not None:
        if inference_device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid inference device {inference_device}")
        device = torch.device(inference_device)
    else:
        device = get_device(cfg)
    use_cuda = device == torch.device("cuda")
    model.to_device(device)

    model_epoch_file = f"model_epoch_{epoch}"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)

    val_score = 0.0
    peak_inference_cpu_mem = 0
    peak_inference_gpu_mem = 0
    tpb = []
    split2loss = {}

    split_losses = {}
    for dataset, split_name in splits:
        desc = "Validation" if split_name == "val" else "Testing"

        tracemalloc.start()

        online = None
        if split_name == "test" and getattr(cfg, "_online", "") and "val" in split_losses:
            online = OnlineUpdater(cfg, model, split_losses["val"], device)

        all_losses = []
        for graphs in dataset:
            for g in log_tqdm(graphs, desc=desc, logging=logging):
                g.to(device=device)

                s = time.time()
                test_fn = test_node_level if (cfg._is_node_level and not cfg._is_hybrid_loss) else test_edge_level
                losses = test_fn(
                    data=g,
                    model=model,
                    split=split_name,
                    model_epoch_file=model_epoch_file,
                    cfg=cfg,
                    device=device,
                )
                all_losses.extend(losses)
                tpb.append(time.time() - s)
                if online is not None:                 # test first, then train
                    online.update(g, losses)

                g.to("cpu")  # Move graph back to CPU to free GPU memory for next batch
                if use_cuda:
                    torch.cuda.empty_cache()

        _, peak_inference_cpu_memory = tracemalloc.get_traced_memory()
        peak_inference_cpu_mem = max(peak_inference_cpu_mem, peak_inference_cpu_memory / (1024**3))
        tracemalloc.stop()

        if use_cuda:
            peak_inference_gpu_memory = torch.cuda.max_memory_allocated(device=device) / (1024**3)
            peak_inference_gpu_mem = max(peak_inference_gpu_mem, peak_inference_gpu_memory)
            torch.cuda.reset_peak_memory_stats(device=device)

        if online is not None:
            online.summary()
        split_losses[split_name] = all_losses
        mean_loss = np.mean(all_losses)
        split2loss[split_name] = mean_loss

        if split_name == "val":
            val_score = model.get_val_ap()
            if logging:
                log(
                    f"[@epoch{epoch:02d}] Validation finished - Val Loss: {mean_loss:.4f}",
                    return_line=True,
                )
        else:
            if logging:
                log(
                    f"[@epoch{epoch:02d}] Test finished - Test Loss: {mean_loss:.4f}",
                    return_line=True,
                )

    del model

    stats = {
        "val_score": val_score,
        "val_loss": split2loss.get("val", None),
        "test_loss": split2loss.get("test", None),
        "peak_inference_cpu_memory": peak_inference_cpu_mem,
        "peak_inference_gpu_memory": peak_inference_gpu_mem,
        "time_per_batch_inference": np.mean(tpb),
    }
    return stats
