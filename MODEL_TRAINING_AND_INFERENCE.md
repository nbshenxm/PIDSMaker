# Model Training, Checkpointing, and Inference Workflow

This guide explains how to train models, save checkpoints, evaluate them, and select the best weights for production use.

## Table of Contents

1. [Quick Answer: Training vs Inference](#quick-answer-training-vs-inference)
2. [Complete Workflow](#complete-workflow)
3. [Training: When and How](#training-when-and-how)
4. [Inference: Cross-Dataset Evaluation](#inference-cross-dataset-evaluation)
5. [Model Selection](#model-selection)
6. [Data Persistence: Your Weights Are Safe](#data-persistence-your-weights-are-safe)
7. [Detailed Examples](#detailed-examples)
8. [Troubleshooting](#troubleshooting)

---

## Quick Answer: Training vs Inference

### When to use the training script (`train_model.sbatch`)

**Use this script ONLY when you want to:**
- Train a new model from scratch
- Retrain an existing model with different hyperparameters
- Generate checkpoint files (model weights) at each epoch

**Example:**
```bash
cd /fs1/projects/benchmarking_PIDS/sherwin_dir/PIDSMaker

# Train Velox on CADETS_E3 (default)
sbatch scripts/train_model.sbatch

# Train Orthrus on THEIA_E5
sbatch scripts/train_model.sbatch orthrus THEIA_E5

# Train Kairos on CADETS_E3
sbatch scripts/train_model.sbatch kairos CADETS_E3
```

### When to use the inference script (`evaluate_all_models.py`)

**Use this script when you want to:**
- Test trained models on different datasets
- Compare performance across models
- Load and use trained weights without retraining
- Generate evaluation metrics

**Example:**
```bash
# Evaluate all trained models
python scripts/evaluate_all_models.py

# Evaluate specific model on specific dataset
python scripts/evaluate_all_models.py --detection-system velox --dataset CADETS_E5

# Use a specific epoch's weights
python scripts/evaluate_all_models.py --detection-system velox --epoch 5
```

### Your model weights are ALWAYS safe

✓ **Old weights are never deleted** - Each training run creates a new checkpoint directory
✓ **Checkpoints persist** - Files saved to `artifacts/training/training/<hash>/<DATASET>/checkpoints/`
✓ **Multiple epochs** - Weights are saved at each epoch, allowing you to compare and select the best
✓ **Metadata tracked** - Each checkpoint includes validation scores and training loss

---

## Complete Workflow

### Phase 1: Train a Model and Generate Checkpoints

```bash
# Step 1: Submit training job (this will take 1-2+ hours)
sbatch scripts/train_model.sbatch velox CADETS_E3

# Step 2: Monitor training progress
tail -f slurm-pids_eval_all-<JOB_ID>.out

# Step 3: Verify checkpoints were saved (after training completes)
ls -la artifacts/training/training/*/CADETS_E3/checkpoints/
```

Expected output:
```
artifacts/training/training/abc123def/CADETS_E3/checkpoints/
├── checkpoints.json          # Metadata about all epochs
├── epoch_0000/
│   ├── state_dict.pkl        # Model weights
│   ├── neighbor_loader.pkl   # TGN-specific state (if applicable)
│   └── memory.pkl            # TGN-specific state (if applicable)
├── epoch_0001/
│   ├── state_dict.pkl
│   ├── neighbor_loader.pkl
│   └── memory.pkl
└── ...
```

### Phase 2: Evaluate Models on Multiple Datasets

```bash
# Run inference on all trained models across all available datasets
python scripts/evaluate_all_models.py

# Or evaluate specific model
python scripts/evaluate_all_models.py --detection-system velox --dataset CADETS_E3

# Check available datasets
python scripts/evaluate_all_models.py --list-datasets-only
```

Output:
```
Available trained models:
  Detection System: velox
  Dataset: CADETS_E3
  Checkpoint dir: artifacts/training/training/abc123def/CADETS_E3/checkpoints
  Epochs available: [0, 1, 2, 3, 4, 5]
  Best validation score: 0.8756
```

### Phase 3: Compare and Select the Best Model

```bash
# Compare all models
python scripts/compare_models.py

# Compare specific system
python scripts/compare_models.py --detection-system velox

# Compare by different metrics
python scripts/compare_models.py --compare-by validation-score    # Default: by validation metric
python scripts/compare_models.py --compare-by cross-dataset        # How model performs across datasets
python scripts/compare_models.py --compare-by inference            # Based on inference results
```

Output includes:
- Ranked list of all models
- Recommended production model
- Checkpoint paths for each model
- How to load the best weights

### Phase 4: Load and Use the Best Model

```python
from pidsmaker.detection.checkpoint_manager import CheckpointManager
from pidsmaker.model import build_model

# Initialize checkpoint manager
manager = CheckpointManager("artifacts/training/training/abc123def/CADETS_E3/checkpoints")

# Load the best checkpoint (automatically selects highest validation score)
model = manager.load_best_checkpoint(
    model_architecture=build_model("velox"),
    device="cuda"
)

# Or load a specific epoch
model = manager.load_checkpoint(
    model_architecture=build_model("velox"),
    epoch=5,
    device="cuda"
)

# Use model for inference
predictions = model(batch_data)
```

---

## Training: When and How

### Script: `scripts/train_model.sbatch`

This SLURM batch script handles the complete training pipeline including:
1. Data loading
2. Model initialization
3. Training loop with validation
4. **Automatic checkpoint saving at each epoch**
5. **Metadata tracking (validation scores, loss, etc.)**

### Making the Script Configurable

The script accepts command-line arguments:

```bash
# Signature
sbatch train_model.sbatch [DETECTION_SYSTEM] [DATASET]

# Examples
sbatch train_model.sbatch velox CADETS_E3        # Defaults
sbatch train_model.sbatch orthrus CADETS_E5      # Different system and dataset
sbatch train_model.sbatch kairos THEIA_E3        # Another combination
sbatch train_model.sbatch magic CLEARSCOPE_E3
```

### Available Detection Systems

- `velox`
- `orthrus` (also: `orthrus_fixed`, `orthrus_non_snooped`)
- `kairos`
- `magic`
- `flash`
- `nodlink`
- `rcaid`
- `threatrace`

### Available Datasets

- `CADETS_E3` (Cyber Advanced Research Defense and Evaluation Test Suite)
- `CADETS_E5`
- `CLEARSCOPE_E3`
- `CLEARSCOPE_E5`
- `FIVEDIRECTIONS_E3`
- `FIVEDIRECTIONS_E5`
- `OPTC_H051`
- `OPTC_H201`
- `OPTC_H501`
- `THEIA_E3`
- `THEIA_E5`
- `TRACE_E3`
- `TRACE_E5`

### Important Flags

#### `--force_restart training`

The script automatically includes this flag to **force retraining** even if the model was already trained:

```bash
python -m pidsmaker.main $DETECTION_SYSTEM $DATASET \
    --artifact_dir ./artifacts/ \
    --database_host localhost \
    --force_restart training  # <-- Forces retraining
```

**Why is this needed?**

PIDSMaker tracks completed tasks with marker files (`done.txt`). When you run the same configuration again:
- ✗ WITHOUT `--force_restart training`: existing checkpoints remain, no retraining
- ✓ WITH `--force_restart training`: training task is restarted, generates new checkpoints

### Checkpoint Configuration

Checkpoints are configured in the YAML config files:

```yaml
# config/velox.yml, config/kairos.yml, etc.
training_loop:
  save_checkpoints: true
  checkpoint_keep_every_n_epochs: 1  # Save all epochs (set to higher values to save disk space)
```

These settings control:
- Whether checkpoints are saved (`save_checkpoints: true/false`)
- How often to save (`checkpoint_keep_every_n_epochs: N`)

---

## SLURM Scripts Overview

You can run all three phases (training, evaluation, comparison) via SLURM batch scripts for cluster execution:

### 1. Training Script: `scripts/train_model.sbatch`

Configurable training with checkpoint saving.

```bash
# Defaults: trains velox on CADETS_E3
sbatch scripts/train_model.sbatch

# Train specific system and dataset
sbatch scripts/train_model.sbatch orthrus CADETS_E5
sbatch scripts/train_model.sbatch kairos THEIA_E3
```

**Resources:**
- Time: 2 hours
- GPU: 1x
- CPU: 8
- Memory: 32GB

### 2. Evaluation Script: `scripts/evaluate_all_models.sbatch`

Run inference on all trained models across datasets.

```bash
# Evaluate all models on all datasets
sbatch scripts/evaluate_all_models.sbatch

# Evaluate specific system
sbatch scripts/evaluate_all_models.sbatch velox

# Evaluate specific system on specific dataset
sbatch scripts/evaluate_all_models.sbatch velox CADETS_E3
```

**Resources:**
- Time: 8 hours (long-running inference)
- GPU: 1x
- CPU: 4
- Memory: 16GB

**Output files:**
- `artifacts/inference_results.json` - Evaluation results for all models
- `slurm-pids_eval_all-<JOB_ID>.out` - Detailed output log

### 3. Comparison Script: `scripts/compare_models.sbatch`

Compare models and select the best checkpoint.

```bash
# Compare by validation score (default)
sbatch scripts/compare_models.sbatch

# Compare specific system
sbatch scripts/compare_models.sbatch velox

# Compare by cross-dataset performance
sbatch scripts/compare_models.sbatch --compare-by cross-dataset

# Compare by inference results
sbatch scripts/compare_models.sbatch --compare-by inference
```

**Resources:**
- Time: 1 hour
- GPU: None
- CPU: 2
- Memory: 8GB

**Output:**
- Ranked list of models
- Production recommendation with checkpoint path
- Model statistics summary

---

### Standalone Inference Script

Once models are trained and checkpoints exist, you can:
1. **Test trained models on any dataset** without retraining
2. **Compare performance** across training/evaluation dataset combinations
3. **Select best weights** based on cross-dataset performance

### Usage

```bash
# 1. Evaluate all trained models on all datasets
python scripts/evaluate_all_models.py

# 2. Evaluate specific model
python scripts/evaluate_all_models.py \
    --detection-system velox \
    --dataset CADETS_E3 \
    --epoch best          # Use best checkpoint, or specific epoch number

# 3. Evaluate on specific evaluation dataset
# (Runs model trained on CADETS_E3 against CADETS_E5)
python scripts/evaluate_all_models.py \
    --detection-system velox \
    --dataset CADETS_E5

# 4. Only list available models without running inference
python scripts/evaluate_all_models.py --list-models-only
```

### Output

Results are saved to `artifacts/inference_results.json`:

```json
[
  {
    "system": "velox",
    "dataset": "CADETS_E3",
    "epoch": 5,
    "success": true,
    "message": "✓ velox on CADETS_E3 (epoch 5): inference completed"
  },
  ...
]
```

---

## Model Selection

### Checkpoint Metadata

Each checkpoint directory contains `checkpoints.json` with metadata:

```json
{
  "detection_system": "velox",
  "best_validation_score": 0.8756,
  "best_epoch": 5,
  "training_loss": 0.1234,
  "total_epochs": 10,
  "epochs": {
    "0": {
      "epoch": 0,
      "validation_score": 0.7123,
      "training_loss": 0.4567,
      "saved_at": "2026-03-20T10:00:00"
    },
    ...
  }
}
```

### Comparison Script

The `compare_models.py` script analyzes checkpoints and recommends the best:

```bash
# Compare all models by validation score
python scripts/compare_models.py

# Compare specific detection system
python scripts/compare_models.py --detection-system velox

# Compare by cross-dataset performance
python scripts/compare_models.py --compare-by cross-dataset

# Compare by inference evaluation results
python scripts/compare_models.py --compare-by inference \
    --results-file artifacts/inference_results.json
```

### Selection Criteria

The recommended model is selected based on:

1. **Highest validation score** - Best performance on training dataset validation set
2. **Cross-dataset generalization** - Model trained on one dataset, tested on others
3. **Training stability** - Consistent loss reduction across epochs
4. **Epoch efficiency** - Best performance reached with reasonable number of epochs

### Example Output

```
========================================================================
RECOMMENDED MODEL (Highest Validation Score)
========================================================================
Detection System: velox
Dataset: CADETS_E3
Best Epoch: 5
Validation Score: 0.8756
Checkpoint Directory: artifacts/training/training/abc123def/CADETS_E3/checkpoints

Model weights: artifacts/training/training/abc123def/CADETS_E3/checkpoints/epoch_0005/state_dict.pkl

To use this model in production:

  from pidsmaker.detection.checkpoint_manager import CheckpointManager
  
  manager = CheckpointManager('artifacts/training/training/abc123def/CADETS_E3/checkpoints')
  
  # Load the best checkpoint
  model = manager.load_best_checkpoint(model_architecture, device='cuda')
```

---

## Data Persistence: Your Weights Are Safe

### How Checkpoints Are Organized

```
artifacts/
└── training/
    └── training/
        ├── abc123def/          # Run 1 (automatically generated hash)
        │   ├── CADETS_E3/
        │   │   └── checkpoints/
        │   │       ├── checkpoints.json
        │   │       ├── epoch_0000/
        │   │       │   └── state_dict.pkl
        │   │       └── epoch_0001/
        │   │           └── state_dict.pkl
        │   └── CADETS_E5/
        │       └── checkpoints/
        │
        └── xyz789abc/          # Run 2 (different hash, completely separate)
            ├── CADETS_E3/
            │   └── checkpoints/
            └── CADETS_E5/
                └── checkpoints/
```

### Key Safety Features

✓ **Isolation by run**: Each training run gets a unique hash directory
✓ **No overwriting**: New training runs create new directories, never delete old ones
✓ **Per-epoch saves**: Multiple checkpoints per run (epoch_0000, epoch_0001, ...)
✓ **Metadata preservation**: Each checkpoint includes validation metrics
✓ **Backward compatibility**: Checkpoints from old runs remain accessible

### Never Lose Weights

Even if you:
1. Train the same model 10 times
2. Train different models
3. Run on different datasets

**All checkpoint files persist** in separate directories. You can:
- Compare all of them
- Load any of them
- Select the best one
- Revert to old ones if needed

---

## Detailed Examples

### Example 1: Train Velox, Compare Across Datasets, Select Best

```bash
# Step 1: Train on CADETS_E3
sbatch scripts/run_velox_cadets_e3.sbatch velox CADETS_E3

# Wait for training to complete (~2 hours), then...

# Step 2: Evaluate on all available datasets
python scripts/evaluate_all_models.py --detection-system velox

# Step 3: Compare performance
python scripts/compare_models.py --detection-system velox --compare-by cross-dataset

# Step 4: View results and select best
cat artifacts/inference_results.json | python -m json.tool

# Step 5: Load the best model
python -c "
from pidsmaker.detection.checkpoint_manager import CheckpointManager
from pidsmaker.model import build_model

manager = CheckpointManager('artifacts/training/training/<HASH>/CADETS_E3/checkpoints')
model = manager.load_best_checkpoint(build_model('velox'), device='cuda')
print('Model loaded successfully!')
"
```

### Example 2: Train Multiple Systems, Find Overall Winner

```bash
# Train multiple systems
sbatch scripts/train_model.sbatch velox CADETS_E3
sbatch scripts/train_model.sbatch orthrus CADETS_E3
sbatch scripts/train_model.sbatch kairos CADETS_E3

# Wait for all jobs to complete, then...

# Compare all systems
python scripts/compare_models.py

# Result: Shows ranked list with Orthrus at top
# Recommended Model:
#   Detection System: orthrus
#   Validation Score: 0.9123
#   Checkpoint: artifacts/training/training/xyz789/CADETS_E3/checkpoints
```

### Example 3: Progressive Training and Evaluation

```bash
# Week 1: Train Velox on CADETS_E3
sbatch scripts/train_model.sbatch velox CADETS_E3

# Week 2: Train Orthrus on same dataset
sbatch scripts/train_model.sbatch orthrus CADETS_E3

# Week 3: Evaluate both models
python scripts/evaluate_all_models.py

# Week 4: Train best model on different dataset
sbatch scripts/train_model.sbatch orthrus CADETS_E5

# Week 5: Final comparison
python scripts/compare_models.py --compare-by cross-dataset

# All original checkpoints from Weeks 1-2 are still available!
```

### Example 4: Load Specific Epoch's Weights

```python
from pidsmaker.detection.checkpoint_manager import CheckpointManager
from pidsmaker.model import build_model

manager = CheckpointManager('artifacts/training/training/abc123/CADETS_E3/checkpoints')

# Option 1: Load best checkpoint automatically
model = manager.load_best_checkpoint(
    model_architecture=build_model('velox'),
    device='cuda'
)

# Option 2: Load specific epoch
model = manager.load_checkpoint(
    model_architecture=build_model('velox'),
    epoch=3,  # Load weights from epoch 3 specifically
    device='cuda'
)

# Option 3: Get info about all checkpoints
checkpoints = manager.list_checkpoints()
for ckpt in checkpoints:
    print(f"Epoch {ckpt['epoch']}: val_score={ckpt['validation_score']}, loss={ckpt['training_loss']}")

# Option 4: Get metadata
info = manager.get_checkpoint_info(epoch=5)
print(f"Epoch 5 validation score: {info['validation_score']}")
```

---

## Troubleshooting

### Issue: Training job fails to start

**Check SLURM compatibility:**
```bash
# Verify you're in the right directory
cd /fs1/projects/benchmarking_PIDS/sherwin_dir/PIDSMaker

# Check conda environment exists
conda activate pids  # Should work without errors

# Verify database is running
psql -U <user> -h localhost -d pids -c "SELECT 1"
```

### Issue: No checkpoints are being saved

**Check if checkpoints are enabled:**
```bash
# Look at the config file
grep -A 5 "training_loop:" config/velox.yml

# Should show:
#   save_checkpoints: true
#   checkpoint_keep_every_n_epochs: 1
```

If `save_checkpoints: false`, enable it:
```bash
sed -i 's/save_checkpoints: false/save_checkpoints: true/' config/velox.yml
```

### Issue: Can't find checkpoints from previous run

**List all checkpoint directories:**
```bash
find artifacts/training/training -type d -name "checkpoints"

# Output shows:
# artifacts/training/training/<hash1>/CADETS_E3/checkpoints
# artifacts/training/training/<hash2>/CADETS_E3/checkpoints
# ...
```

**Note:** Each SLURM job creates a unique hash. Use the hash from your training job's output.

### Issue: Training job already marked as done, won't retrain

**Force restart training task:**
```bash
# The script already includes this, but if needed:
python -m pidsmaker.main velox CADETS_E3 \
    --artifact_dir ./artifacts/ \
    --database_host localhost \
    --force_restart training
```

### Issue: Previous checkpoints got deleted

**They shouldn't be!** Each new training run uses a new hash directory. If you're not seeing old checkpoints:

```bash
# List all training runs
ls -la artifacts/training/training/

# Each subdirectory is a separate run with its own checkpoints
# If old ones are missing, check if you explicitly deleted them
```

---

## Summary: Training vs Inference Lifetime

| Phase | Tool | When | Output | Frequency |
|-------|------|------|--------|-----------|
| **Training** | `sbatch scripts/run_velox_cadets_e3.sbatch` | Need new weights | Checkpoints at each epoch | Once per config/hyperparameter |
| **Evaluation** | `python scripts/evaluate_all_models.py` | Test on new dataset | Inference results JSON | Before selecting model |
| **Comparison** | `python scripts/compare_models.py` | Pick best model | Rankings and recommendation | After evaluation |
| **Production** | `CheckpointManager.load_best_checkpoint()` | Serve predictions | Loaded model weights | Continuously |

**Remember:** Training creates and saves checkpoints. Inference and comparison use those existing checkpoints. Your old weights are always safe!
