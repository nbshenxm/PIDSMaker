#!/bin/bash
#SBATCH --job-name=pids_train_model
#SBATCH --output=slurm-pids_train_model-%j.out
#SBATCH --error=slurm-pids_train_model-%j.err
#SBATCH --nodelist=cn01
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#
# Usage: sbatch train_model.sbatch [DETECTION_SYSTEM] [DATASET] [TRAIN_SEED] [FORCE_RESTART]
# Examples:
#   sbatch train_model.sbatch            # Uses defaults: velox CADETS_E3
#   sbatch train_model.sbatch orthrus CADETS_E5
#   sbatch train_model.sbatch kairos THEIA_E3
#   sbatch train_model.sbatch velox CADETS_E3 42   # seed tracked in run metadata
#   sbatch train_model.sbatch velox CADETS_E3      # reuse existing path (may skip done tasks)
#   sbatch train_model.sbatch velox CADETS_E3 42 training   # force rerun training and downstream tasks
#
# Available detection systems: velox, orthrus, kairos, magic, flash, nodlink, rcaid, threatrace
# Available datasets: CADETS_E3, CADETS_E5, CLEARSCOPE_E3, CLEARSCOPE_E5, FIVEDIRECTIONS_E3, FIVEDIRECTIONS_E5, OPTC_H051, OPTC_H201, OPTC_H501, THEIA_E3, THEIA_E5, TRACE_E3, TRACE_E5

DETECTION_SYSTEM="${1:-velox}"
DATASET="${2:-CADETS_E3}"
TRAIN_SEED="${3:-}"
FORCE_RESTART="${4:-}"
export DETECTION_SYSTEM DATASET TRAIN_SEED FORCE_RESTART

set -euo pipefail

# Resolve repository root.
# In SLURM, scripts are often copied to a spool dir, so BASH_SOURCE points there.
# Prefer SLURM_SUBMIT_DIR (where sbatch was invoked), then fall back to script location.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/pidsmaker" ]]; then
    REPO_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if [[ ! -d "${REPO_DIR}/pidsmaker" && -d "/fs1/projects/benchmarking_PIDS/sherwin_dir/PIDSMaker/pidsmaker" ]]; then
    REPO_DIR="/fs1/projects/benchmarking_PIDS/sherwin_dir/PIDSMaker"
fi

if [[ ! -d "${REPO_DIR}/pidsmaker" ]]; then
    echo "ERROR: could not resolve repository root containing pidsmaker package"
    echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    echo "Resolved REPO_DIR=${REPO_DIR}"
    exit 1
fi

cd "${REPO_DIR}"

# Activate conda environment
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found in PATH"
    exit 1
fi

# Initialize conda for non-interactive shells
# shellcheck disable=SC1091
set +u
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pids
set -u

echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "Repo dir: ${REPO_DIR}"
echo "Detection system: ${DETECTION_SYSTEM}"
echo "Dataset: ${DATASET}"
echo "Train seed: ${TRAIN_SEED:-<config default>}"
echo "Force restart: ${FORCE_RESTART:-<none>}"
echo "Database host: localhost (apptainer postgres)"
echo "Python: $(which python)"

# Ensure local package is importable even if SLURM launches from another cwd context.
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
echo "Working dir: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"

if [[ ! -d "${REPO_DIR}/pidsmaker" ]]; then
    echo "ERROR: package directory not found at ${REPO_DIR}/pidsmaker"
    exit 1
fi

if [[ ! -d "${REPO_DIR}/scripts/apptainer" ]]; then
    echo "ERROR: missing apptainer folder at ${REPO_DIR}/scripts/apptainer"
    exit 1
fi

# Ensure container runtime is available on compute node.
if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load apptainer >/dev/null 2>&1 || true
        module load singularity >/dev/null 2>&1 || true
    fi
fi

if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
    echo "ERROR: neither apptainer nor singularity is available on this node"
    echo "Hint: load a container module before submitting, e.g."
    echo "  module load apptainer"
    exit 1
fi

echo ""
echo "=== Starting Apptainer Postgres ==="
(
    cd "${REPO_DIR}/scripts/apptainer"
    make up
)

# Train the model with the specified detection system and dataset.
# Behavior:
# - With TRAIN_SEED: set the seed and record it in metadata
# - Without TRAIN_SEED: reuse existing task path; completed tasks may be skipped
# - With FORCE_RESTART: rerun the named task and downstream tasks, e.g. training
echo ""
echo "=== Starting training for ${DETECTION_SYSTEM} on ${DATASET} ==="

TRAIN_CMD=(
    python -m pidsmaker.main
    "${DETECTION_SYSTEM}"
    "${DATASET}"
    --artifact_dir ./artifacts/
    --database_host localhost
)

if [[ -n "${TRAIN_SEED}" ]]; then
    TRAIN_CMD+=(--training.seed "${TRAIN_SEED}")
else
    echo "No TRAIN_SEED provided: reusing existing run hash (no forced retrain)."
fi

if [[ -n "${FORCE_RESTART}" ]]; then
    TRAIN_CMD+=(--force_restart "${FORCE_RESTART}")
fi

echo "Command: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

echo ""
echo "=== Writing training run metadata ==="
python - <<'PY'
import json
import os
from datetime import datetime, timezone

from pidsmaker.config import get_runtime_required_args, get_yml_cfg

detection_system = os.environ["DETECTION_SYSTEM"]
dataset = os.environ["DATASET"]
train_seed = os.environ.get("TRAIN_SEED", "")
force_restart = os.environ.get("FORCE_RESTART", "")

args_list = [
    detection_system,
    dataset,
    "--artifact_dir",
    "./artifacts/",
    "--database_host",
    "localhost",
]
if train_seed:
    args_list += ["--training.seed", train_seed]
if force_restart:
    args_list += ["--force_restart", force_restart]

args = get_runtime_required_args(args=args_list)
cfg = get_yml_cfg(args)

metadata = {
    "detection_system": detection_system,
    "dataset": dataset,
    "effective_training_seed": cfg.training.seed,
    "input_train_seed": train_seed if train_seed else None,
    "force_restart": force_restart if force_restart else None,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}

metadata_path = os.path.join(cfg.training._task_path, "run_metadata.json")
os.makedirs(cfg.training._task_path, exist_ok=True)
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
    f.write("\n")

print(f"Wrote metadata: {metadata_path}")
PY

echo ""
echo "=== Training completed ==="
echo "Checkpoint path for this run:"
python - <<'PY'
import os

from pidsmaker.config import get_runtime_required_args, get_yml_cfg

detection_system = os.environ["DETECTION_SYSTEM"]
dataset = os.environ["DATASET"]
train_seed = os.environ.get("TRAIN_SEED", "")
force_restart = os.environ.get("FORCE_RESTART", "")

args_list = [
    detection_system,
    dataset,
    "--artifact_dir",
    "./artifacts/",
    "--database_host",
    "localhost",
]
if train_seed:
    args_list += ["--training.seed", train_seed]
if force_restart:
    args_list += ["--force_restart", force_restart]

args = get_runtime_required_args(args=args_list)
cfg = get_yml_cfg(args)

checkpoints_dir = os.path.join(cfg.training._task_path, "checkpoints")
print(checkpoints_dir)
PY

echo "Finished at: $(date)"
