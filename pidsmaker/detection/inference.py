"""Standalone inference script for loading and testing model checkpoints.

This module allows you to:
1. Load a trained model checkpoint from a specific epoch
2. Run inference on any dataset (validation or test)
3. Generate predictions and anomaly scores
4. Save results to CSV

Usage:
    python -m pidsmaker.detection.inference \
        --config velox \
        --dataset CADETS_E5 \
        --checkpoint-epoch 5 \
        --split test \
        --output results.csv
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import wandb

from pidsmaker.config import DATASET_DEFAULT_CONFIG, get_runtime_required_args, get_yml_cfg
from pidsmaker.detection.checkpoint_manager import CheckpointManager
from pidsmaker.detection.training_methods import inference_loop
from pidsmaker.factory import build_model
from pidsmaker.pids.velox.model import VeloxModel
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils.utils import get_device, log, log_start, set_seed


def load_checkpoint_model(
    cfg,
    checkpoint_dir: str,
    epoch: int,
    device,
) -> Tuple[object, int]:
    """Load a model from checkpoint.
    
    Args:
        cfg: Configuration object
        checkpoint_dir: Directory containing checkpoints
        epoch: Epoch number to load (or -1 for best)
        device: Device to load model to
        
    Returns:
        Tuple of (model, loaded_epoch)
    """
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_every_n=1,
        cfg=cfg,
    )
    
    # Get list of available checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    log(f"Available checkpoints: {sorted(checkpoints.keys())}", return_line=True)
    
    # Determine which epoch to load
    if epoch == -1:
        # Load best checkpoint
        loaded_epoch = int(checkpoint_manager.metadata.get("best_checkpoint", -1))
        if loaded_epoch == -1:
            raise ValueError("No best checkpoint found")
        log(f"Loading best checkpoint: epoch {loaded_epoch}", return_line=True)
    else:
        loaded_epoch = epoch
        if loaded_epoch not in checkpoints:
            raise ValueError(
                f"Epoch {loaded_epoch} not found in checkpoints. Available: {sorted(checkpoints.keys())}"
            )
        log(f"Loading checkpoint: epoch {loaded_epoch}", return_line=True)
    
    # Build model architecture
    train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(cfg)
    if getattr(cfg, "_model", None) == "velox":
        sample = train_data[0][0]
        in_dim = sample.x_src.shape[1]
        model = VeloxModel(
            cfg=cfg,
            in_dim=in_dim,
            out_dim=cfg.dataset.num_node_types,
            device=device,
        ).to(device)
    else:
        model = build_model(
            data_sample=train_data[0][0], device=device, cfg=cfg, max_node_num=max_node_num
        )
    
    # Load checkpoint weights
    checkpoint_manager.load_checkpoint(model, loaded_epoch)
    model.to(device)
    model.eval()
    
    return model, loaded_epoch


def run_inference(
    cfg,
    model,
    epoch: int,
    dataset_split: str = "test",
    output_dir: Optional[str] = None,
) -> dict:
    """Run inference on a dataset split.
    
    Args:
        cfg: Configuration object
        model: Model to run inference with
        epoch: Epoch number (for logging/labeling outputs)
        dataset_split: Which split to evaluate ("val" or "test")
        output_dir: Directory to save results (optional)
        
    Returns:
        dict: Inference statistics
    """
    log_start(__file__)
    set_seed(cfg)
    
    device = get_device(cfg)
    
    # Load data
    train_data, val_data, test_data, _ = get_preprocessed_graphs(cfg)
    
    # Map split name to data
    if dataset_split == "val":
        eval_data = val_data
    elif dataset_split == "test":
        eval_data = test_data
    else:
        raise ValueError(f"Invalid dataset split: {dataset_split}. Must be 'val' or 'test'")
    
    # Run inference
    log(f"Running inference on {dataset_split} split...", return_line=True)
    stats = inference_loop.main(
        cfg=cfg,
        model=model,
        val_data=val_data,
        test_data=test_data,
        epoch=epoch,
        split=dataset_split,
        logging=True,
    )
    
    log(f"Inference complete. Results:", return_line=True)
    for key, value in stats.items():
        if isinstance(value, float):
            log(f"  {key}: {value:.4f}", return_line=True)
        else:
            log(f"  {key}: {value}", return_line=True)
    
    return stats


def main():
    """Main entry point for standalone inference."""
    parser = argparse.ArgumentParser(
        description="Run inference using a saved model checkpoint"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Model config name (e.g., velox, orthrus, kairos)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset name (e.g., CADETS_E3, CADETS_E5)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing checkpoints (default: <task_path>/checkpoints)",
    )
    parser.add_argument(
        "--epoch",
        "-e",
        type=int,
        default=-1,
        help="Epoch to load (-1 for best checkpoint, default: -1)",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        help="Optional path to write inference metrics JSON",
    )
    
    args = parser.parse_args()

    # Allow case-insensitive dataset names (e.g., OPTC_H051 -> optc_h051).
    available_datasets = list(DATASET_DEFAULT_CONFIG.keys())
    lower_to_dataset = {name.lower(): name for name in available_datasets}
    normalized_dataset = lower_to_dataset.get(args.dataset.lower())
    if normalized_dataset is None:
        raise ValueError(
            f"Unknown dataset {args.dataset}. Available datasets are {available_datasets}"
        )
    args.dataset = normalized_dataset
    
    # Build full cfg from model + dataset to ensure task paths and data loaders are correct.
    runtime_args = get_runtime_required_args(
        args=[
            args.config,
            args.dataset,
            "--artifact_dir",
            "./artifacts/",
            "--database_host",
            "localhost",
        ]
    )
    cfg = get_yml_cfg(runtime_args)
    set_seed(cfg)
    device = get_device(cfg)
    
    # Initialize Weights & Biases if enabled
    if not args.no_wandb:
        wandb.init(
            project=cfg.experiment.project if hasattr(cfg, 'experiment') else "pidsmaker",
            name=f"inference_epoch_{args.epoch}",
            config={
                "config": args.config,
                "epoch": args.epoch,
                "split": args.split,
                "dataset": cfg.dataset.name,
            },
        )
    
    try:
        # Determine checkpoint directory
        checkpoint_dir = args.checkpoint_dir or os.path.join(
            cfg.training._task_path, "checkpoints"
        )
        
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Load model from checkpoint
        log(f"Loading checkpoint from: {checkpoint_dir}", return_line=True)
        model, loaded_epoch = load_checkpoint_model(
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
            epoch=args.epoch,
            device=device,
        )
        
        # Run inference
        stats = run_inference(
            cfg=cfg,
            model=model,
            epoch=loaded_epoch,
            dataset_split=args.split,
            output_dir=args.output_dir,
        )
        
        # Log final results
        if not args.no_wandb:
            wandb.log(stats)
            wandb.finish()

        if args.metrics_out:
            metrics_path = Path(args.metrics_out)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "config": args.config,
                "dataset": args.dataset,
                "split": args.split,
                "epoch": loaded_epoch,
                "checkpoint_dir": checkpoint_dir,
                "stats": stats,
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
        
        log("Inference complete!", return_line=True)
        return stats
        
    except Exception as e:
        log(f"Error during inference: {e}", return_line=True)
        if not args.no_wandb:
            wandb.finish(1)
        raise


if __name__ == "__main__":
    main()
