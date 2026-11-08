# SLURM Batch Scripts Quick Reference

## Overview

Three sbatch scripts are available for running the complete training, evaluation, and comparison workflow on the cluster:

| Script | Purpose | Time | GPU | Command |
|--------|---------|------|-----|---------|
| `train_model.sbatch` | Train a model and save checkpoints | 2h | 1x | `sbatch scripts/train_model.sbatch [SYSTEM] [DATASET]` |
| `evaluate_all_models.sbatch` | Run inference on all trained models | 8h | 1x | `sbatch scripts/evaluate_all_models.sbatch [SYSTEM] [DATASET]` |
| `compare_models.sbatch` | Compare models and select best | 1h | None | `sbatch scripts/compare_models.sbatch [OPTIONS]` |

## 1. Training: `train_model.sbatch`

### Basic Usage

```bash
cd /fs1/projects/benchmarking_PIDS/sherwin_dir/PIDSMaker

# Train Velox on CADETS_E3 (defaults)
sbatch scripts/train_model.sbatch

# Train specific system and dataset
sbatch scripts/train_model.sbatch orthrus CADETS_E5
sbatch scripts/train_model.sbatch kairos THEIA_E3
```

### Arguments

- `DETECTION_SYSTEM` (optional): System to train (velox, orthrus, kairos, magic, flash, nodlink, rcaid, threatrace). Default: `velox`
- `DATASET` (optional): Dataset to use (CADETS_E3, CADETS_E5, THEIA_E3, THEIA_E5, etc.). Default: `CADETS_E3`

### Output Files

```
slurm-pids_<SYSTEM>_<DATASET>-<JOB_ID>.out      # Main output log
slurm-pids_<SYSTEM>_<DATASET>-<JOB_ID>.err      # Error log
artifacts/training/training/<HASH>/<DATASET>/checkpoints/  # Saved checkpoints
```

### Monitor Progress

```bash
# Watch the job running in real-time
tail -f slurm-pids_<SYSTEM>_<DATASET>-<JOB_ID>.out

# Check job status
squeue -u $USER
scontrol show job <JOB_ID>

# After completion, verify checkpoints exist
ls -la artifacts/training/training/*/CADETS_E3/checkpoints/
```

---

## 2. Evaluation: `evaluate_all_models.sbatch`

### Basic Usage

```bash
# Evaluate all trained models on all datasets
sbatch scripts/evaluate_all_models.sbatch

# Evaluate specific detection system
sbatch scripts/evaluate_all_models.sbatch velox

# Evaluate specific system on specific dataset
sbatch scripts/evaluate_all_models.sbatch velox CADETS_E3
```

### Arguments

- `DETECTION_SYSTEM` (optional): Filter evaluation to specific system. Omit to evaluate all.
- `DATASET` (optional): Filter evaluation to specific dataset. Omit to evaluate all.

### Output Files

```
slurm-pids_eval_all-<JOB_ID>.out                # Main output log
artifacts/inference_results.json                # Evaluation results (JSON)
```

### Example: Full Workflow

```bash
# Step 1: Train model
sbatch scripts/train_model.sbatch velox CADETS_E3
# Wait ~2 hours...

# Step 2: Evaluate on all datasets
sbatch scripts/evaluate_all_models.sbatch velox
# Wait ~8 hours...

# Step 3: View results
cat artifacts/inference_results.json | python -m json.tool
```

### Results Format

```json
[
  {
    "system": "velox",
    "dataset": "CADETS_E3",
    "epoch": 5,
    "success": true,
    "message": "✓ velox on CADETS_E3 (epoch 5): inference completed"
  },
  {
    "system": "velox",
    "dataset": "CADETS_E5",
    "epoch": 5,
    "success": true,
    "message": "✓ velox on CADETS_E5 (epoch 5): inference completed"
  }
]
```

---

## 3. Comparison: `compare_models.sbatch`

### Basic Usage

```bash
# Compare all models by validation score
sbatch scripts/compare_models.sbatch

# Compare specific detection system only
sbatch scripts/compare_models.sbatch velox

# Compare by different metric
sbatch scripts/compare_models.sbatch --compare-by cross-dataset
sbatch scripts/compare_models.sbatch --compare-by inference
```

### Arguments

- `DETECTION_SYSTEM` (optional): Filter comparison to specific system
- `--compare-by` (optional): Comparison metric
  - `validation-score` (default): Rank by best validation metric
  - `cross-dataset`: Show performance across datasets
  - `inference`: Compare inference evaluation results

### Output Files

```
slurm-pids_compare_models-<JOB_ID>.out          # Output log with rankings
```

### Output Example

```
================================================================================
Model Comparison by Validation Metrics (from checkpoint metadata)
================================================================================

Rank  System       Dataset         Val Score    Best Epoch  Checkpoint Dir
---------------------------------
1     orthrus      CADETS_E3       0.9123       5           checkpoints
2     velox        CADETS_E3       0.8756       4           checkpoints
3     kairos       CADETS_E3       0.8234       3           checkpoints

================================================================================
RECOMMENDED MODEL (Highest Validation Score)
================================================================================
Detection System: orthrus
Dataset: CADETS_E3
Best Epoch: 5
Validation Score: 0.9123
Checkpoint Directory: artifacts/training/training/abc123def/CADETS_E3/checkpoints

Model weights: artifacts/training/training/abc123def/CADETS_E3/checkpoints/epoch_0005/state_dict.pkl
```

---

## Complete Workflow Example

```bash
#!/bin/bash
# Example: Train multiple systems and compare

cd /fs1/projects/benchmarking_PIDS/sherwin_dir/PIDSMaker

# Phase 1: Train models (parallel)
echo "Submitting training jobs..."
JOB1=$(sbatch --parsable scripts/train_model.sbatch velox CADETS_E3)
JOB2=$(sbatch --parsable scripts/train_model.sbatch orthrus CADETS_E3)
JOB3=$(sbatch --parsable scripts/train_model.sbatch kairos CADETS_E3)

echo "Training jobs submitted: $JOB1, $JOB2, $JOB3"

# Phase 2: Wait for training to complete
squeue --job=$JOB1,$JOB2,$JOB3 --format="%i %T" | grep -q "RUNNING" && echo "Waiting for training..."
squeue --job=$JOB1,$JOB2,$JOB3 --format=%T | tail -3

# Phase 3: Evaluate all models
echo "Evaluating models..."
sbatch scripts/evaluate_all_models.sbatch

# Phase 4: Compare and get recommendation
echo "Comparing models..."
sbatch scripts/compare_models.sbatch
```

---

## Common Operations

### Check All Running SLURM Jobs

```bash
squeue -u $USER
```

### Cancel a Job

```bash
scancel <JOB_ID>
```

### View Job Output After Completion

```bash
# View last N lines
tail -100 slurm-pids_*.out

# View with timestamps
cat slurm-pids_*.out | less
```

### Get Training Statistics

```bash
# Find how long training took
grep "Started at\|Ended at" slurm-pids_*train_model*.out

# Check if checkpoint files were saved correctly
grep "Checkpoint files found:" slurm-pids_*train_model*.out -A 20
```

### Load Best Checkpoint After Comparison

Once you have run `compare_models.sbatch`, you'll get output showing the recommended checkpoint path. Load it with:

```python
from pidsmaker.detection.checkpoint_manager import CheckpointManager
from pidsmaker.model import build_model

# Path from the comparison output
checkpoint_dir = "artifacts/training/training/abc123def/CADETS_E3/checkpoints"

manager = CheckpointManager(checkpoint_dir)
model = manager.load_best_checkpoint(
    model_architecture=build_model("orthrus"),
    device="cuda"
)

# Use the model
predictions = model(test_data)
```

---

## Troubleshooting

### Job Won't Start: "No nodes available"

```bash
# Check node availability
sinfo | head -20

# Try a different node (update SBATCH --nodelist in the script)
```

### Out of Memory

Increase the memory request in the sbatch script:
- Training: increase `--mem=32G` to `48G` or `64G`
- Evaluation: increase `--mem=16G` to `32G`
- Comparison: usually not an issue

### Conda Environment Not Found

```bash
# Verify conda environment exists
conda env list | grep pids

# If missing, create it
conda create -n pids python=3.9
conda activate pids
pip install -r requirements.txt
```

### GPU Out of Memory During Training

Reduce batch size in config, or resubmit to a different GPU:
```bash
#SBATCH --gres=gpu:1          # Single GPU (current)
#SBATCH --gres=gpu:a100       # Specific GPU type (if available)
```

---

## Performance Expectations

| Phase | Typical Duration | Bottleneck |
|-------|------------------|-----------|
| Training (1 epoch) | 10-30 min | GPU computation |
| Full training (4-10 epochs) | 1-2 hours | GPU computation |
| Evaluation (1 model, 1 dataset) | 5-30 min | GPU memory/compute |
| Evaluation (5 models, 13 datasets) | 4-8 hours | 65 inferences, batched |
| Comparison | < 5 min | JSON parsing |

---

## Quick Reference Commands

```bash
# Submit and track a training job
JOB=$(sbatch --parsable scripts/train_model.sbatch velox CADETS_E3)
watch -n 5 "squeue --job=$JOB"

# Evaluate all trained models
sbatch scripts/evaluate_all_models.sbatch

# Get final recommendation
sbatch scripts/compare_models.sbatch

# Check all recent SLURM outputs
ls -ltr slurm-*.out | tail -10

# View latest job log
tail -50 slurm-*.out | sort | tail -50
```
