"""
Example script showing how to use the new checkpoint and inference features.

This script demonstrates:
1. Training with checkpoint saving
2. Loading checkpoints
3. Running inference on different datasets
4. Comparing results across epochs
"""

import os
from pathlib import Path

from pidsmaker.config import get_yml_cfg
from pidsmaker.detection.checkpoint_manager import CheckpointManager
from pidsmaker.detection.inference import load_checkpoint_model, run_inference
from pidsmaker.utils.utils import get_device, log, set_seed


def example_list_checkpoints():
    """Example: List all available checkpoints."""
    log("=" * 60, return_line=True)
    log("Example 1: Listing Available Checkpoints", return_line=True)
    log("=" * 60, return_line=True)
    
    cfg = get_yml_cfg("config/default.yml")
    checkpoint_dir = os.path.join(cfg.training._task_path, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        log(f"No checkpoints found at {checkpoint_dir}", return_line=True)
        return
    
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir, cfg=cfg)
    checkpoints = manager.list_checkpoints()
    
    log(f"\nFound {len(checkpoints)} checkpoints:", return_line=True)
    for epoch, info in sorted(checkpoints.items()):
        val_score = info.get("val_score", "N/A")
        train_loss = info.get("train_loss", "N/A")
        log(f"  Epoch {epoch:4d}: val_score={val_score:7.4f}, train_loss={train_loss:7.4f}", return_line=True)
    
    log(f"\nBest checkpoint: Epoch {manager.metadata['best_checkpoint']}", return_line=True)
    log("")


def example_load_and_test():
    """Example: Load a checkpoint and run inference."""
    log("=" * 60, return_line=True)
    log("Example 2: Load Checkpoint and Run Inference", return_line=True)
    log("=" * 60, return_line=True)
    
    cfg = get_yml_cfg("config/default.yml")
    device = get_device(cfg)
    set_seed(cfg)
    checkpoint_dir = os.path.join(cfg.training._task_path, "checkpoints")
    
    try:
        # Load best checkpoint
        log("\nLoading best checkpoint...", return_line=True)
        model, loaded_epoch = load_checkpoint_model(
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
            epoch=-1,  # -1 means best
            device=device,
        )
        log(f"Successfully loaded checkpoint from epoch {loaded_epoch}", return_line=True)
        
        # Run inference
        log("\nRunning inference on test set...", return_line=True)
        stats = run_inference(
            cfg=cfg,
            model=model,
            epoch=loaded_epoch,
            dataset_split="test",
        )
        
        log("\nInference Results:", return_line=True)
        log(f"  Val Loss: {stats['val_loss']:.4f}", return_line=True)
        log(f"  Test Loss: {stats['test_loss']:.4f}", return_line=True)
        log(f"  Val Score: {stats['val_score']:.4f}", return_line=True)
        log("")
        
    except Exception as e:
        log(f"Error: {e}", return_line=True)


def example_compare_epochs():
    """Example: Load and test multiple epochs."""
    log("=" * 60, return_line=True)
    log("Example 3: Compare Results Across Epochs", return_line=True)
    log("=" * 60, return_line=True)
    
    cfg = get_yml_cfg("config/default.yml")
    device = get_device(cfg)
    set_seed(cfg)
    checkpoint_dir = os.path.join(cfg.training._task_path, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        log(f"No checkpoints found at {checkpoint_dir}", return_line=True)
        return
    
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir, cfg=cfg)
    checkpoints = sorted(manager.list_checkpoints().keys())
    
    if len(checkpoints) < 2:
        log("Need at least 2 checkpoints for comparison", return_line=True)
        return
    
    # Compare first and last checkpoint
    epochs_to_test = [checkpoints[0], checkpoints[-1]]
    
    log(f"\nComparing epochs: {epochs_to_test}", return_line=True)
    log("\nEpoch | Val Loss | Test Loss | Val Score", return_line=True)
    log("------|----------|-----------|----------", return_line=True)
    
    try:
        for epoch in epochs_to_test:
            model, _ = load_checkpoint_model(
                cfg=cfg,
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                device=device,
            )
            
            stats = run_inference(
                cfg=cfg,
                model=model,
                epoch=epoch,
                dataset_split="test",
                output_dir=None,
            )
            
            log(
                f"{epoch:5d} | {stats['val_loss']:8.4f} | {stats['test_loss']:9.4f} | {stats['val_score']:9.4f}",
                return_line=True,
            )
        
        log("")
        
    except Exception as e:
        log(f"Error: {e}", return_line=True)


def example_load_specific_epoch():
    """Example: Load a specific epoch."""
    log("=" * 60, return_line=True)
    log("Example 4: Load Specific Epoch", return_line=True)
    log("=" * 60, return_line=True)
    
    cfg = get_yml_cfg("config/default.yml")
    device = get_device(cfg)
    checkpoint_dir = os.path.join(cfg.training._task_path, "checkpoints")
    
    epoch_to_load = 5
    
    try:
        log(f"\nLoading checkpoint from epoch {epoch_to_load}...", return_line=True)
        model, loaded_epoch = load_checkpoint_model(
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
            epoch=epoch_to_load,
            device=device,
        )
        log(f"Successfully loaded epoch {loaded_epoch}", return_line=True)
        log("")
        
    except Exception as e:
        log(f"Error: {e}", return_line=True)


if __name__ == "__main__":
    log("", return_line=True)
    log("Checkpoint and Inference Usage Examples", return_line=True)
    log("=" * 60, return_line=True)
    log("", return_line=True)
    
    # Run examples
    example_list_checkpoints()
    # example_load_and_test()
    # example_compare_epochs()
    # example_load_specific_epoch()
    
    log("Examples complete!", return_line=True)
