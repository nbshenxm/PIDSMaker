#!/bin/bash
#SBATCH --job-name=pids-db
#SBATCH --output=logs/slurm/db_server-%j.out
#SBATCH --error=logs/slurm/db_server-%j.out
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# ONE long-lived PostgreSQL server for all PIDSMaker jobs.
#
# Why: the per-job `make up` in the old train_model.sbatch starts a postgres on
# whatever node the job lands on, against the SAME data dir on /fs1. Two such
# jobs on different nodes = two postmasters on one data dir = corruption risk.
# Here a single server runs; every pipeline job connects over the network using
# the host/port written to $DB_INFO.
#
#   sbatch scripts/caliber/db_server.sbatch      # start (once)
#   cat  "$DB_INFO"                              # -> "<host> <port>"
#   scancel <jobid>                              # stop (fast shutdown)
set -euo pipefail

PG_ROOT="${PG_ROOT:-/fs1/projects/benchmarking_PIDS/sherwin_dir/PIDSMaker/scripts/apptainer}"
PG_PORT="${PG_PORT:-54329}"
DB_INFO="${DB_INFO:-/fs1/projects/benchmarking_PIDS/sherwin_dir/db_server.info}"

cd "${PG_ROOT}"
module load apptainer >/dev/null 2>&1 || true
CONTAINER_CMD="$(command -v apptainer || command -v singularity)"

# Refuse to start if a postmaster may already own the data dir.
if [[ -f postgres_data/postmaster.pid ]]; then
    if [[ -f "${DB_INFO}" ]]; then
        read -r OTHER_HOST OTHER_PORT OTHER_JOB < "${DB_INFO}" || true
        if [[ -n "${OTHER_JOB:-}" ]] && squeue -h -j "${OTHER_JOB}" 2>/dev/null | grep -q .; then
            echo "ERROR: db server job ${OTHER_JOB} is still running on ${OTHER_HOST}:${OTHER_PORT}"
            exit 1
        fi
    fi
    echo "WARNING: stale postmaster.pid found (no live db job). Contents:"
    cat postgres_data/postmaster.pid || true
    echo "Make sure no other postgres (e.g. an old train_model.sbatch job) uses ${PG_ROOT}/postgres_data."
    echo "Remove postgres_data/postmaster.pid manually once verified, then resubmit."
    exit 1
fi

mkdir -p postgres_run postgres_log
export APPTAINERENV_POSTGRES_PASSWORD=postgres
export APPTAINERENV_POSTGRES_USER=postgres
export APPTAINERENV_POSTGRES_DB=postgres

BIND="--bind postgres_data:/var/lib/postgresql/data --bind postgres_run:/var/run/postgresql --bind postgres_log:/var/log"

stop_pg() {
    echo "[$(date)] stopping postgres (fast)"
    "${CONTAINER_CMD}" exec ${BIND} postgres.sif pg_ctl -D /var/lib/postgresql/data -m fast stop || true
    rm -f "${DB_INFO}"
}
trap stop_pg TERM INT EXIT

echo "[$(date)] starting postgres on $(hostname):${PG_PORT}"
"${CONTAINER_CMD}" exec ${BIND} postgres.sif \
    docker-entrypoint.sh postgres -p "${PG_PORT}" -c listen_addresses='*' \
    -c max_connections=200 -c shared_buffers=8GB -c work_mem=256MB &
PG_PID=$!

for _ in $(seq 1 60); do
    if "${CONTAINER_CMD}" exec ${BIND} postgres.sif pg_isready -h localhost -p "${PG_PORT}" -U postgres >/dev/null 2>&1; then
        echo "$(hostname) ${PG_PORT} ${SLURM_JOB_ID}" > "${DB_INFO}"
        echo "[$(date)] postgres ready; wrote ${DB_INFO}: $(cat "${DB_INFO}")"
        break
    fi
    sleep 2
done
[[ -f "${DB_INFO}" ]] || { echo "postgres failed to start"; exit 1; }

wait "${PG_PID}"
