# Training/Inference Separation - Implementation Summary

## What Was Done

I successfully refactored PIDSMaker to separate training and inference, allowing you to:
- ✓ Save model weights at each epoch automatically
- ✓ Load trained models from specific epochs
- ✓ Run inference independently on any dataset
- ✓ Test the same model on multiple datasets without retraining

## Files Created (3 new files)

### 1. `pidsmaker/detection/checkpoint_manager.py`
A complete checkpoint management system with methods for:
- **Saving**: `save_checkpoint(model, epoch, val_score, train_loss)`
- **Loading**: `load_checkpoint(model, epoch)` or `load_best_checkpoint(model)`
- **Listing**: `list_checkpoints()` - view all available checkpoints
- **Management**: `delete_checkpoint(epoch)`, `get_checkpoint_info(epoch)`
- **Metadata**: Tracks validation scores and training loss per epoch

### 2. `pidsmaker/detection/inference.py`
Standalone inference script for testing without retraining:
- **Python API**: `load_checkpoint_model()`, `run_inference()`
- **Command-line**: Run inference from terminal
- **Flexible dataset support**: Test on any dataset using checkpoints from training

### 3. `CHECKPOINT_AND_INFERENCE.md`
Comprehensive user guide with:
- Configuration instructions
- API reference for CheckpointManager
- 5+ working examples
- Best practices and troubleshooting
- Architecture overview

## Files Modified (2 files)

### 1. `pidsmaker/config/config.py`
Added to `EXPERIMENTS_CONFIG.training_loop`:
```yaml
save_checkpoints: bool                      # Enable checkpoint saving
checkpoint_keep_every_n_epochs: int        # Save every Nth epoch (1=all)
```

### 2. `pidsmaker/detection/training_methods/training_loop.py`
- Added CheckpointManager initialization
- Integrated automatic checkpoint saving after validation
- Saves with epoch, validation score, and training loss metadata
- Falls back to checkpoint loading when available

## Quick Start

### 1. Enable Checkpoints in Config
```yaml
training_loop:
  save_checkpoints: true
  checkpoint_keep_every_n_epochs: 1  # Save all epochs
```

### 2. Train Normally
```bash
python -m pidsmaker.main --config config.yml --dataset THEIA_E5
```
Checkpoints are automatically saved to: `<task_path>/checkpoints/`

### 3. Test on Different Dataset
```bash
# List available checkpoints
python -c "
from pidsmaker.detection.checkpoint_manager import CheckpointManager
m = CheckpointManager('path/to/checkpoints')
for epoch, info in m.list_checkpoints().items():
    print(f'Epoch {epoch}: val_score={info[\"val_score\"]:.4f}')
"

# Run inference on a checkpoint
python -m pidsmaker.detection.inference \
    --config config.yml \
    --epoch 5 \
    --split test
```

### 4. Load in Code
```python
from pidsmaker.detection.inference import load_checkpoint_model

model, epoch = load_checkpoint_model(
    cfg=cfg,
    checkpoint_dir="path/to/checkpoints",
    epoch=-1,  # -1 = best checkpoint
    device=device
)

# Use model for inference
with torch.no_grad():
    predictions = model(data_batch)
```

## Key Features

- **Automatic saving**: No code changes needed - just set config option
- **Smart filtering**: Save only every Nth epoch to save disk space
- **Metadata tracking**: Val scores and training loss recorded for each checkpoint
- **Best model selection**: Automatically tracks and makes best checkpoint easy to load
- **TGN support**: Handles TGN-specific state (neighbor_loader, memory)
- **Standalone inference**: No training code needed to use saved models
- **CLI interface**: Run inference from command line without Python scripts
- **Flexible datasets**: Load checkpoints from one dataset, test on another

## Directory Structure After Training

```
<task_path>/
└── checkpoints/
    ├── epoch_0000/
    │   ├── state_dict.pkl          # Model weights
    │   ├── neighbor_loader.pkl     # TGN components (if applicable)
    │   └── memory.pkl
    ├── epoch_0001/
    │   ├── state_dict.pkl
    │   ├── neighbor_loader.pkl
    │   └── memory.pkl
    └── checkpoints.json            # Metadata (epochs, scores, best epoch)
```

## Configuration Example

```yaml
# config.yml
training_loop:
  run_evaluation: best_epoch
  save_checkpoints: true           # Enable checkpoints
  checkpoint_keep_every_n_epochs: 1 # Save all epochs (or 5 to save every 5th)

training:
  num_epochs: 20
  patience: 3
  # ... rest of training config
```

## Next Steps

1. **Enable checkpoints** in your config: Add `save_checkpoints: true`
2. **Train a model** normally - checkpoints auto-save
3. **Read the guide** at `CHECKPOINT_AND_INFERENCE.md` for detailed usage
4. **Run examples** from `examples/checkpoint_and_inference_examples.py`
5. **Test on different datasets** using saved checkpoints

## Backward Compatibility

- ✓ Checkpoints are **disabled by default** - no breaking changes
- ✓ Existing training code works unchanged
- ✓ Optional feature that requires explicit config to enable
- ✓ Can coexist with current pipeline

## Questions or Issues?

Check `CHECKPOINT_AND_INFERENCE.md` for:
- API Reference (CheckpointManager class)
- Troubleshooting section
- Best practices
- Architecture overview
