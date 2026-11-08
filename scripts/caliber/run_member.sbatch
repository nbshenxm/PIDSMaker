#!/bin/bash
#SBATCH --job-name=cal-member
#SBATCH --output=logs/slurm/member-%x-%A_%a.out
#SBATCH --error=logs/slurm/member-%x-%A_%a.out
#SBATCH --qos=long
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#
# Train + evaluate ONE ensemble member (one seed) of a PIDS on a dataset.
# Seed = BASE_SEED + SLURM_ARRAY_TASK_ID (or BASE_SEED when not an array job).
# Connects to the shared postgres started by db_server.sbatch.
#
#   sbatch [--array=1-9] scripts/caliber/run_member.sbatch SYSTEM DATASET [extra pidsmaker args...]
# Prefer scripts/caliber/submit_ensemble.sh, which handles the shared-artifact race.
#
# Env overrides: BASE_SEED (default 0), ARTIFACT_DIR, DB_INFO, CONDA_ENV.
set -euo pipefail

SYSTEM="$1"; DATASET="$2"; shift 2
EXTRA_ARGS=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/fs1/projects/benchmarking_PIDS/sherwin_dir/artifacts_v3/}"
DB_INFO="${DB_INFO:-/fs1/projects/benchmarking_PIDS/sherwin_dir/db_server.info}"
CONDA_ENV="${CONDA_ENV:-pids3}"
BASE_SEED="${BASE_SEED:-0}"
# SEEDS (space-separated) runs several members sequentially in one job, which
# keeps the number of submitted jobs low (the account has a submit limit).
if [[ -n "${SEEDS:-}" ]]; then
    read -r -a SEED_LIST <<< "${SEEDS}"
else
    SEED_LIST=( $(( BASE_SEED + ${SLURM_ARRAY_TASK_ID:-0} )) )
fi

cd "${REPO_DIR}"
[[ -f "${DB_INFO}" ]] || { echo "ERROR: ${DB_INFO} missing - start scripts/caliber/db_server.sbatch first"; exit 1; }
read -r DB_HOST DB_PORT DB_JOB < "${DB_INFO}"

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u
# The env's libstdc++ must win over /lib64's (nltk -> sqlite3 -> ICU needs CXXABI_1.3.15).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
# Determinism knobs used by upstream scripts/run.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export NLTK_DATA="${NLTK_DATA:-${HOME}/nltk_data}"
export CALIBER_PACK=1   # pack edge-loss CSVs after each member (file-count quota)

echo "host=$(hostname) system=${SYSTEM} dataset=${DATASET} seeds=${SEED_LIST[*]} db=${DB_HOST}:${DB_PORT}"
echo "extra args: ${EXTRA_ARGS[*]:-<none>}"
echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

for SEED in "${SEED_LIST[@]}"; do
    echo "=== member start: seed=${SEED} $(date)"
    python pidsmaker/main.py "${SYSTEM}" "${DATASET}" \
        --artifact_dir "${ARTIFACT_DIR}" \
        --database_host "${DB_HOST}" --database_port "${DB_PORT}" \
        --training.seed "${SEED}" \
        "${EXTRA_ARGS[@]}"
    # Record where this member's outputs live (consumed by caliber.data).
    python -m caliber.pidsmaker_bridge manifest "${SYSTEM}" "${DATASET}" \
        --artifact_dir "${ARTIFACT_DIR}" --training.seed "${SEED}" "${EXTRA_ARGS[@]}"
    echo "=== member done: ${SYSTEM} ${DATASET} seed=${SEED} $(date)"
done
