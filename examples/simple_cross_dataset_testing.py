"""
Simple example: Load a trained model and test on a different dataset

This script demonstrates the most common use case:
1. Train on THEIA_E5
2. Save checkpoints
3. Load checkpoint and test on CADETS_E5 (or any other dataset)
"""

import torch

from pidsmaker.config import get_yml_cfg
from pidsmaker.detection.checkpoint_manager import CheckpointManager
from pidsmaker.detection.inference import load_checkpoint_model, run_inference
from pidsmaker.utils.utils import get_device, log, set_seed


def test_on_different_dataset():
    """Load a checkpoint from training and test on a different dataset."""
    
    # IMPORTANT: Use the config from the TRAINING dataset
    # This ensures the model architecture matches what was trained
    training_config_path = "config/kairos.yml"
    checkpoint_dir = "path/to/trained/model/checkpoints"
    
    log("=" * 70, return_line=True)
    log("Cross-Dataset Testing Example", return_line=True)
    log("=" * 70, return_line=True)
    
    # Setup
    cfg = get_yml_cfg(training_config_path)
    set_seed(cfg)
    device = get_device(cfg)
    
    log(f"\nTraining dataset config: {training_config_path}", return_line=True)
    log(f"Checkpoint directory: {checkpoint_dir}", return_line=True)
    
    # Step 1: List available checkpoints
    log("\n[Step 1] Listing available checkpoints...", return_line=True)
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir, cfg=cfg)
    
    checkpoints = manager.list_checkpoints()
    if not checkpoints:
        log("ERROR: No checkpoints found!", return_line=True)
        return
    
    log(f"Found {len(checkpoints)} checkpoints:", return_line=True)
    for epoch, info in sorted(checkpoints.items())[:5]:  # Show first 5
        log(
            f"  Epoch {epoch:3d}: val_score={info['val_score']:7.4f}, "
            f"train_loss={info['train_loss']:7.4f}",
            return_line=True,
        )
    if len(checkpoints) > 5:
        log(f"  ... and {len(checkpoints) - 5} more", return_line=True)
    
    best_epoch = int(manager.metadata["best_checkpoint"])
    best_score = manager.metadata["best_val_score"]
    log(f"\nBest checkpoint: Epoch {best_epoch} (val_score={best_score:.4f})", return_line=True)
    
    # Step 2: Load best checkpoint
    log("\n[Step 2] Loading best checkpoint...", return_line=True)
    try:
        model, loaded_epoch = load_checkpoint_model(
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
            epoch=-1,  # -1 = best
            device=device,
        )
        log(f"Successfully loaded epoch {loaded_epoch}", return_line=True)
    except Exception as e:
        log(f"ERROR loading checkpoint: {e}", return_line=True)
        return
    
    # Step 3: Run inference
    log("\n[Step 3] Running inference on test set...", return_line=True)
    try:
        stats = run_inference(
            cfg=cfg,
            model=model,
            epoch=loaded_epoch,
            dataset_split="test",  # or "val"
        )
        
        log("\n[Results]", return_line=True)
        log(f"  Val Loss:    {stats['val_loss']:.4f}", return_line=True)
        log(f"  Test Loss:   {stats['test_loss']:.4f}", return_line=True)
        log(f"  Val Score:   {stats['val_score']:.4f}", return_line=True)
        log(f"  Val CPU Mem: {stats['peak_inference_cpu_memory']:.2f} GB", return_line=True)
        log(f"  Val GPU Mem: {stats['peak_inference_gpu_memory']:.2f} GB", return_line=True)
        
    except Exception as e:
        log(f"ERROR during inference: {e}", return_line=True)
        return
    
    log("\n" + "=" * 70, return_line=True)
    log("Test complete!", return_line=True)


def compare_multiple_epochs():
    """Load multiple checkpoints and compare results."""
    
    training_config_path = "config/kairos.yml"
    checkpoint_dir = "path/to/trained/model/checkpoints"
    
    log("=" * 70, return_line=True)
    log("Compare Multiple Epochs", return_line=True)
    log("=" * 70, return_line=True)
    
    cfg = get_yml_cfg(training_config_path)
    set_seed(cfg)
    device = get_device(cfg)
    
    # Get checkpoints
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir, cfg=cfg)
    checkpoints = sorted(manager.list_checkpoints().keys())
    
    if len(checkpoints) < 2:
        log("Need at least 2 checkpoints to compare", return_line=True)
        return
    
    # Test specific epochs
    epochs_to_test = [checkpoints[0], checkpoints[len(checkpoints)//2], checkpoints[-1]]
    
    log(f"\nTesting epochs: {epochs_to_test}\n", return_line=True)
    log("Epoch | Val Loss | Test Loss | Val Score", return_line=True)
    log("------|----------|-----------|----------", return_line=True)
    
    for epoch in epochs_to_test:
        try:
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
            )
            
            log(
                f"{epoch:5d} | {stats['val_loss']:8.4f} | "
                f"{stats['test_loss']:9.4f} | {stats['val_score']:9.4f}",
                return_line=True,
            )
            
            # Clean up to save memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            log(f"Error at epoch {epoch}: {e}", return_line=True)
    
    log("=" * 70, return_line=True)


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    test_on_different_dataset()
    
    # compare_multiple_epochs()
