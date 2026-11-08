#!/bin/bash
#SBATCH --job-name=pids_eval_all
#SBATCH --output=slurm-pids_eval_all-%j.out
#SBATCH --error=slurm-pids_eval_all-%j.err
#SBATCH --exclude=gpu7
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --qos=long
#SBATCH --requeue
#
# Usage: sbatch evaluate_all_models.sbatch [OPTIONS]
# Examples:
#   sbatch evaluate_all_models.sbatch                    # Best model per train set on all test datasets
#   sbatch evaluate_all_models.sbatch velox              # Only velox models
#   sbatch evaluate_all_models.sbatch velox CADETS_E5    # Only velox models, test on CADETS_E5
#
# Available detection systems: velox, orthrus, kairos, magic, flash, nodlink, rcaid, threatrace
# Available datasets: CADETS_E3, CADETS_E5, CLEARSCOPE_E3, CLEARSCOPE_E5, etc.

DETECTION_SYSTEM="${1:-}"
TEST_DATASET="${2:-}"

set -euo pipefail

# Resolve repository root
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
set +u
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pids
set -u

echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "Repo dir: ${REPO_DIR}"
echo "Python: $(which python)"
echo "SLURM job id: ${SLURM_JOB_ID:-<unset>}"
echo "SLURM mem request (node): ${SLURM_MEM_PER_NODE:-<unset>}"

# Ensure local package is importable
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

# Build evaluation command
EVAL_CMD="python -u scripts/evaluate_all_models.py --prepare-timeout-seconds 0 --inference-timeout-seconds 0"

if [[ -n "${DETECTION_SYSTEM}" ]]; then
    EVAL_CMD="${EVAL_CMD} --detection-system ${DETECTION_SYSTEM}"
fi

if [[ -n "${TEST_DATASET}" ]]; then
    EVAL_CMD="${EVAL_CMD} --test-dataset ${TEST_DATASET}"
fi

echo ""
echo "=== Starting model evaluation ==="
echo "Command: ${EVAL_CMD}"
echo ""

${EVAL_CMD}

echo ""
echo "=== Evaluation completed ==="
echo "Results saved to: artifacts/inference_results.json"
echo "Ended at: $(date)"
