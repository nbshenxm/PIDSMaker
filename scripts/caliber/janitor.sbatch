#!/bin/bash
#SBATCH --job-name=cal-janitor
#SBATCH --output=logs/slurm/janitor-%j.out
#SBATCH --error=logs/slurm/janitor-%j.out
#SBATCH --qos=long
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#
# Unattended queue maintenance. Every INTERVAL_MIN minutes, failed tasks whose log shows a
# TRANSIENT cause (database restart / connection loss, disk quota) are moved back to pending,
# at most MAX_RETRY times per task. Any other failure stays in failed/ for inspection.
set -uo pipefail
QUEUE="${QUEUE:-/fs1/projects/benchmarking_PIDS/sherwin_dir/caliber_queue}"
INTERVAL_MIN="${INTERVAL_MIN:-30}"
MAX_RETRY="${MAX_RETRY:-3}"
TRANSIENT='OperationalError|could not connect to server|server closed the connection|Connection refused|Disk quota exceeded'
mkdir -p "${QUEUE}/retries"

while true; do
    for f in "${QUEUE}"/failed/*; do
        [[ -e "${f}" ]] || continue
        name=$(basename "${f}")
        log="${QUEUE}/logs/${name}.log"
        grep -qE "${TRANSIENT}" "${log}" 2>/dev/null || continue
        n=$(cat "${QUEUE}/retries/${name}" 2>/dev/null || echo 0)
        (( n >= MAX_RETRY )) && continue
        echo $((n + 1)) > "${QUEUE}/retries/${name}"
        mv "${f}" "${QUEUE}/pending/${name}" && echo "[$(date)] requeued ${name} (retry $((n + 1)))"
    done
    echo "[$(date)] pending $(ls "${QUEUE}/pending" | wc -l) running $(ls "${QUEUE}/running" | wc -l)" \
         "done $(ls "${QUEUE}/done" | wc -l) failed $(ls "${QUEUE}/failed" | wc -l)"
    sleep $(( INTERVAL_MIN * 60 ))
done
