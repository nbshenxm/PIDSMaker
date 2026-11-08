#!/bin/bash
#SBATCH --job-name=cal-worker
#SBATCH --output=logs/slurm/worker-%j.out
#SBATCH --error=logs/slurm/worker-%j.out
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#
# GPU worker: repeatedly claims a task from the CALIBER queue and runs it.
# A task is claimed by atomically renaming pending/<name> -> running/<name>.<jobid>,
# so concurrent workers never run the same task. Tasks whose REQUIRES is not yet
# in done/ are skipped until it is. The worker exits when nothing is claimable
# for IDLE_EXIT_MIN minutes, or when too little wall time remains.
#
#   sbatch scripts/caliber/worker.sbatch          # start one worker (start a few)
#
# Env: QUEUE, ARTIFACT_DIR, DB_INFO, CONDA_ENV, IDLE_EXIT_MIN (default 20),
#      MIN_REMAINING_MIN (default 240: do not start a task with less time left).
set -uo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
QUEUE="${QUEUE:-/fs1/projects/benchmarking_PIDS/sherwin_dir/caliber_queue}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/fs1/projects/benchmarking_PIDS/sherwin_dir/artifacts_v3/}"
DB_INFO="${DB_INFO:-/fs1/projects/benchmarking_PIDS/sherwin_dir/db_server.info}"
CONDA_ENV="${CONDA_ENV:-pids3}"
IDLE_EXIT_MIN="${IDLE_EXIT_MIN:-20}"
MIN_REMAINING_MIN="${MIN_REMAINING_MIN:-240}"
JOB="${SLURM_JOB_ID:-local$$}"

cd "${REPO_DIR}"
mkdir -p "${QUEUE}"/{pending,running,done,failed,logs}

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export NLTK_DATA="${NLTK_DATA:-${HOME}/nltk_data}"
export CALIBER_PACK=1   # pack edge-loss CSVs after each member (file-count quota)
python -c "import torch; print('worker', '${JOB}', 'on', '$(hostname)', 'cuda:', torch.cuda.is_available())"

START=$(date +%s)
LIMIT_MIN=$(squeue -h -j "${JOB}" -o %l 2>/dev/null | awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print 2880 }')
LIMIT_MIN="${LIMIT_MIN:-2880}"

remaining_min() { echo $(( LIMIT_MIN - ( $(date +%s) - START ) / 60 )); }

claim() {  # prints claimed path or nothing
    local f name req
    for f in $(ls -1tr "${QUEUE}/pending" 2>/dev/null | grep -v '^\.'); do   # FIFO
        req=$(grep -m1 '^REQUIRES=' "${QUEUE}/pending/${f}" 2>/dev/null | cut -d= -f2-)
        if [[ -n "${req}" ]] && ! compgen -G "${QUEUE}/done/${req}*" >/dev/null; then
            continue
        fi
        if mv "${QUEUE}/pending/${f}" "${QUEUE}/running/${f}.${JOB}" 2>/dev/null; then
            echo "${QUEUE}/running/${f}.${JOB}"
            return
        fi
    done
}

idle_since=$(date +%s)
while true; do
    if (( $(remaining_min) < MIN_REMAINING_MIN )); then
        echo "[$(date)] less than ${MIN_REMAINING_MIN} min left; exiting"; break
    fi
    task=$(claim)
    if [[ -z "${task}" ]]; then
        if (( ( $(date +%s) - idle_since ) / 60 >= IDLE_EXIT_MIN )); then
            echo "[$(date)] idle for ${IDLE_EXIT_MIN} min; exiting"; break
        fi
        sleep 60; continue
    fi
    idle_since=$(date +%s)

    name=$(basename "${task}" ".${JOB}")
    SYSTEM=$(grep -m1 '^SYSTEM=' "${task}" | cut -d= -f2-)
    DATASET=$(grep -m1 '^DATASET=' "${task}" | cut -d= -f2-)
    SEED=$(grep -m1 '^SEED=' "${task}" | cut -d= -f2-)
    ARGS=$(grep -m1 '^ARGS=' "${task}" | cut -d= -f2-)
    read -r DB_HOST DB_PORT _ < "${DB_INFO}"
    log="${QUEUE}/logs/${name}.log"
    echo "[$(date)] START ${name} (log ${log})"

    # shellcheck disable=SC2086
    if python pidsmaker/main.py "${SYSTEM}" "${DATASET}" \
            --artifact_dir "${ARTIFACT_DIR}" \
            --database_host "${DB_HOST}" --database_port "${DB_PORT}" \
            --training.seed "${SEED}" ${ARGS} > "${log}" 2>&1 \
       && python -m caliber.pidsmaker_bridge manifest "${SYSTEM}" "${DATASET}" \
            --artifact_dir "${ARTIFACT_DIR}" --training.seed "${SEED}" ${ARGS} >> "${log}" 2>&1; then
        mv "${task}" "${QUEUE}/done/${name}"
        echo "[$(date)] DONE  ${name}"
    else
        mv "${task}" "${QUEUE}/failed/${name}"
        echo "[$(date)] FAIL  ${name} -> see ${log}"
    fi
done
