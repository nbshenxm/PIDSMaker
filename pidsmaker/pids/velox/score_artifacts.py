"""Velox edge score artifact writer.

This records the original edge-level Velox detector outputs in a more inspectable
form without changing the detector logic.
"""

import json
import os

import pandas as pd
import torch
import torch.nn.functional as F

from pidsmaker.utils.utils import log, log_tqdm


def _threshold_from_scores(scores, threshold_method):
    if not scores:
        return 0.0, "empty"

    method = threshold_method.strip()
    series = pd.Series(scores, dtype=float)

    if method == "max_val_loss":
        return float(series.max()), method
    if method == "mean_val_loss":
        return float(series.mean()), method
    if method == "nodlink":
        return float(series.quantile(0.90)), method
    if method == "magic":
        return float(series.mean()), method

    return float(series.mean()), f"{method}:mean_fallback"


def _threshold_method(cfg):
    eval_method = getattr(cfg.evaluation, "used_method", "")
    if eval_method == "edge_evaluation":
        return cfg.evaluation.edge_evaluation.threshold_method
    return cfg.evaluation.node_evaluation.threshold_method


@torch.no_grad()
def _collect_split_records(model, dataset, split_name, device):
    records = []
    model.eval()
    model.reset_state()

    validation = split_name == "val"
    for graphs in dataset:
        for data in log_tqdm(graphs, desc=f"Velox edge scores {split_name}", logging=False):
            data.to(device=device)
            results = model(data, inference=True, validation=validation)

            scores = results["anomaly_score"].detach().cpu()
            logits = results["raw_logits"].detach().cpu()
            log_probs = results["out"].detach().cpu()
            probs = F.softmax(logits, dim=1)
            pred = results["predicted_edge_type"].detach().cpu()
            target = results["target_edge_type"].detach().cpu()

            edge_index = data.original_edge_index.detach().cpu()
            srcnodes = edge_index[0]
            dstnodes = edge_index[1]
            times = data.t.detach().cpu()
            edge_types = (data.edge_type.max(dim=1).indices + 1).detach().cpu()

            top_prob = probs.max(dim=1).values
            top_logit = logits.max(dim=1).values
            observed_prob = probs[torch.arange(len(target)), target]
            observed_logit = logits[torch.arange(len(target)), target]
            observed_log_prob = log_probs[torch.arange(len(target)), target]

            for idx in range(len(scores)):
                records.append(
                    {
                        "split": split_name,
                        "srcnode": int(srcnodes[idx].item()),
                        "dstnode": int(dstnodes[idx].item()),
                        "time": int(times[idx].item()),
                        "edge_type": int(edge_types[idx].item()),
                        "target_edge_type": int(target[idx].item()),
                        "predicted_edge_type": int(pred[idx].item()),
                        "correct_prediction": int(target[idx].item() == pred[idx].item()),
                        "anomaly_score": float(scores[idx].item()),
                        "loss": float(scores[idx].item()),
                        "observed_probability": float(observed_prob[idx].item()),
                        "observed_log_probability": float(observed_log_prob[idx].item()),
                        "observed_logit": float(observed_logit[idx].item()),
                        "predicted_probability": float(top_prob[idx].item()),
                        "predicted_logit": float(top_logit[idx].item()),
                        "logit_margin": float(top_logit[idx].item() - observed_logit[idx].item()),
                    }
                )

            data.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return records


def write_epoch_score_artifacts(cfg, model, val_data, test_data, epoch, device):
    """Write raw Velox edge scores plus validation threshold diagnostics."""
    split_to_records = {
        "val": _collect_split_records(model, val_data, "val", device),
        "test": _collect_split_records(model, test_data, "test", device),
    }

    val_scores = [record["anomaly_score"] for record in split_to_records["val"]]
    threshold_method = _threshold_method(cfg)
    threshold, threshold_source = _threshold_from_scores(val_scores, threshold_method)

    out_dir = os.path.join(cfg.training._task_path, "velox_scores", f"model_epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)

    for split_name, records in split_to_records.items():
        for record in records:
            record["threshold"] = threshold
            record["threshold_source"] = threshold_source
            record["distance_to_threshold"] = record["anomaly_score"] - threshold
            record["above_threshold"] = int(record["anomaly_score"] >= threshold)

        pd.DataFrame.from_records(records).to_csv(
            os.path.join(out_dir, f"{split_name}.csv"),
            index=False,
        )

    summary = {
        "epoch": epoch,
        "detector_level": "edge",
        "threshold": threshold,
        "threshold_source": threshold_source,
        "validation_score_count": len(val_scores),
        "validation_score_mean": float(pd.Series(val_scores, dtype=float).mean())
        if val_scores
        else None,
        "validation_score_max": float(max(val_scores)) if val_scores else None,
        "files": {
            "val": os.path.join(out_dir, "val.csv"),
            "test": os.path.join(out_dir, "test.csv"),
        },
    }
    summary_path = os.path.join(out_dir, "threshold_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    log(
        f"[@epoch{epoch:02d}] Velox edge score artifacts written to {out_dir} "
        f"(threshold={threshold:.6f}, source={threshold_source})",
        return_line=True,
    )
    return summary
