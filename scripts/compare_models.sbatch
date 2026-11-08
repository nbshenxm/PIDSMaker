#!/bin/bash
#SBATCH --job-name=pids_compare_models
#SBATCH --output=slurm-pids_compare_models-%j.out
#SBATCH --error=slurm-pids_compare_models-%j.err
#SBATCH --nodelist=cn01
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=long
#
# Usage: sbatch compare_models.sbatch [OPTIONS]
# Examples:
#   sbatch compare_models.sbatch                         # Compare all models by validation score
#   sbatch compare_models.sbatch velox                   # Compare only velox models
#   sbatch compare_models.sbatch --compare-by cross-dataset
#   sbatch compare_models.sbatch --compare-by inference
#
# Comparison modes: validation-score (default), cross-dataset, inference

DETECTION_SYSTEM="${1:-}"
COMPARE_BY="${2:-validation-score}"

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

# Ensure local package is importable
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
echo "Working dir: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"

if [[ ! -d "${REPO_DIR}/pidsmaker" ]]; then
    echo "ERROR: package directory not found at ${REPO_DIR}/pidsmaker"
    exit 1
fi

# Build comparison command
COMPARE_CMD="python scripts/compare_models.py"

if [[ -n "${DETECTION_SYSTEM}" && "${DETECTION_SYSTEM}" != "validation-score" && "${DETECTION_SYSTEM}" != "cross-dataset" && "${DETECTION_SYSTEM}" != "inference" ]]; then
    COMPARE_CMD="${COMPARE_CMD} --detection-system ${DETECTION_SYSTEM}"
fi

if [[ -n "${COMPARE_BY}" ]]; then
    COMPARE_CMD="${COMPARE_CMD} --compare-by ${COMPARE_BY}"
fi

echo ""
echo "=== Starting model comparison and analysis ==="
echo "Command: ${COMPARE_CMD}"
echo ""

${COMPARE_CMD}

echo ""
echo "=== Comparison completed ==="
echo "Results summary saved above."
echo "Ended at: $(date)"
