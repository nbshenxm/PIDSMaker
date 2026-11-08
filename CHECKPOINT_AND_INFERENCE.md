# Training and Inference Separation Guide

This document explains the new checkpoint and inference features that allow separating model training from inference, enabling you to save model weights at each epoch and load them to test on different datasets.

## Overview

Previously, training and inference were coupled together in a single pipeline. Now you can:

1. **Save model checkpoints** at each epoch during training
2. **Load trained models** from specific epochs
3. **Run inference independently** on any dataset without retraining
4. **Test on multiple datasets** using the same model weights

## Configuration

### Enable Checkpoint Saving

In your configuration YAML file, add these settings to the `training_loop` section:

```yaml
training_loop:
  run_evaluation: best_epoch  # or "each_epoch"
  save_checkpoints: true      # Enable checkpoint saving
  checkpoint_keep_every_n_epochs: 1  # 1 = save all epochs, 5 = save every 5th epoch
```

### Directory Structure

When checkpoints are enabled, model weights are saved to:
```
<task_path>/checkpoints/
├── epoch_0000/
│   ├── state_dict.pkl          # Model weights
│   ├── neighbor_loader.pkl     # TGN-specific (if using TGN encoder)
│   └── memory.pkl              # TGN memory state (if applicable)
├── epoch_0001/
│   ├── state_dict.pkl
│   ├── neighbor_loader.pkl
│   └── memory.pkl
└── checkpoints.json            # Metadata about all checkpoints
```

## Usage Examples

### 1. Training with Checkpoint Saving

Use the standard training pipeline with checkpoint saving enabled:

```bash
python -m pidsmaker.main \
    --config config.yml \
    --dataset THEIA_E5
```

The training will automatically save checkpoints according to your configuration. You'll see logs like:

```
Checkpoint saved: epoch 0 | val_score: 0.8234 | path: /path/to/checkpoints/epoch_0000
Checkpoint saved: epoch 1 | val_score: 0.8456 | path: /path/to/checkpoints/epoch_0001
```

### 2. List Available Checkpoints

```python
from pidsmaker.detection.checkpoint_manager import CheckpointManager

checkpoint_manager = CheckpointManager(checkpoint_dir="/path/to/checkpoints")
checkpoints = checkpoint_manager.list_checkpoints()

for epoch, info in sorted(checkpoints.items()):
    print(f"Epoch {epoch}: val_score={info['val_score']:.4f}, "
          f"train_loss={info['train_loss']:.4f}")
```

Output:
```
Epoch 0: val_score=0.8234, train_loss=0.1523
Epoch 1: val_score=0.8456, train_loss=0.1245
Epoch 2: val_score=0.8512, train_loss=0.1089
```

### 3. Load a Specific Checkpoint and Run Inference

```python
from pidsmaker.config import get_yml_cfg
from pidsmaker.detection.checkpoint_manager import CheckpointManager
from pidsmaker.factory import build_model
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils.utils import get_device, set_seed

# Load configuration
cfg = get_yml_cfg("config.yml")
set_seed(cfg)
device = get_device(cfg)

# Build model architecture
train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(cfg)
model = build_model(
    data_sample=train_data[0][0],
    device=device,
    cfg=cfg,
    max_node_num=max_node_num
)

# Load checkpoint from epoch 5
checkpoint_manager = CheckpointManager(
    checkpoint_dir="path/to/checkpoints"
)
checkpoint_manager.load_checkpoint(model, epoch=5)

# Now use model for inference
model.eval()
with torch.no_grad():
    results = model(data_batch)
```

### 4. Test on Different Datasets

Load a trained checkpoint and test on a different dataset:

```python
from pidsmaker.config import get_yml_cfg
from pidsmaker.detection.inference import load_checkpoint_model, run_inference

# Load config for the original training dataset
cfg = get_yml_cfg("config_theia_e5.yml")

# Load a checkpoint from previous training
model, loaded_epoch = load_checkpoint_model(
    cfg=cfg,
    checkpoint_dir="path/to/checkpoints",
    epoch=5,  # or -1 for best
    device=device,
)

# Change dataset configuration (if needed)
cfg.dataset.name = "CADETS_E5"  # Test on different dataset

# Run inference
stats = run_inference(
    cfg=cfg,
    model=model,
    epoch=loaded_epoch,
    dataset_split="test",
)

print(f"Test Loss: {stats['test_loss']:.4f}")
print(f"Val Score: {stats['val_score']:.4f}")
```

### 5. Standalone Inference Script

Run inference from the command line without modifying code:

```bash
# Load best checkpoint and test
python -m pidsmaker.detection.inference \
    --config config.yml \
    --epoch -1 \
    --split test

# Load specific epoch
python -m pidsmaker.detection.inference \
    --config config.yml \
    --epoch 5 \
    --split test \
    --output-dir ./results

# Use custom checkpoint directory
python -m pidsmaker.detection.inference \
    --config config.yml \
    --checkpoint-dir /path/to/custom/checkpoints \
    --epoch 3 \
    --split val
```

## CheckpointManager API Reference

### Initialization

```python
from pidsmaker.detection.checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="/path/to/checkpoints",
    keep_every_n=1,  # Save all epochs
    cfg=cfg            # Configuration object
)
```

### Save Checkpoint

```python
checkpoint_path = manager.save_checkpoint(
    model=model,
    epoch=0,
    val_score=0.8234,
    train_loss=0.1523,
)
```

### Load Checkpoint

```python
# Load specific epoch
success = manager.load_checkpoint(model, epoch=5)

# Load best checkpoint
best_epoch = manager.load_best_checkpoint(model)
```

### List Checkpoints

```python
checkpoints = manager.list_checkpoints()
# Returns: {epoch: {metadata}, ...}

# Get info for specific epoch
info = manager.get_checkpoint_info(epoch=5)
```

### Delete Checkpoint

```python
manager.delete_checkpoint(epoch=5)
```

## Best Practices

1. **Set `checkpoint_keep_every_n_epochs` wisely**:
   - Use `1` for smaller datasets (saves all epochs)
   - Use `5` or `10` for larger datasets (saves disk space)

2. **Use best checkpoint for production**:
   ```python
   manager.load_best_checkpoint(model)
   ```

3. **Save configuration alongside checkpoints**:
   - The checkpoint metadata automatically records val_score and train_loss
   - Always keep the config.yml used for training

4. **Clean up old checkpoints** if disk space is limited:
   ```python
   for epoch in range(0, 100, 5):  # Keep only every 5th
       if epoch % 5 != 0:
           manager.delete_checkpoint(epoch)
   ```

## Troubleshooting

### No checkpoints found

- Check that `save_checkpoints: true` in your config
- Verify the checkpoint directory path is correct
- Training must have completed at least one cycle

### Out of memory when loading checkpoint

- Reduce batch size in validation
- Increase `checkpoint_keep_every_n_epochs` to save fewer checkpoints
- Consider using a different device (CPU vs GPU)

### Model performance different when loaded from checkpoint

- Ensure the same configuration is used for both training and inference
- Check that model is in `eval()` mode for inference
- Some components like dropout and batch norm behave differently in train vs eval mode

## Architecture Changes

The refactoring introduced:

1. **CheckpointManager** (`pidsmaker/detection/checkpoint_manager.py`): Handles all checkpoint save/load operations
2. **Updated training_loop.py**: Now saves checkpoints after each epoch evaluation
3. **Standalone inference.py** (`pidsmaker/detection/inference.py`): Allows running inference without retraining
4. **Config extensions**: New `training_loop` configuration options for checkpoint control

## Migration Guide (if upgrading from old version)

If you have existing trained models without checkpoints:

1. Retrain with checkpoint saving enabled to get epoch-level checkpoints
2. Or manually save best model using CheckpointManager
3. All new trainings will automatically save checkpoints

The old training pipeline still works, but checkpoint saving is now optional and disabled by default.
