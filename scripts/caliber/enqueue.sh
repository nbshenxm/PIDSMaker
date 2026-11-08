#!/bin/bash
# Enqueue an M-member seed ensemble into the CALIBER task queue (no sbatch).
# Workers (scripts/caliber/worker.sbatch) pick tasks up. Member 0 is a
# prerequisite of members 1..M-1 because it builds the shared upstream
# artifacts (construction ... batching) that later members reuse.
#
#   scripts/caliber/enqueue.sh SYSTEM DATASET M [extra pidsmaker args...]
#   e.g. scripts/caliber/enqueue.sh orthrus CLEARSCOPE_E3 10
#        scripts/caliber/enqueue.sh velox THEIA_E3 10 --training.stable_optim=True
#
# Env: QUEUE (default /fs1/projects/benchmarking_PIDS/sherwin_dir/caliber_queue),
#      BASE_SEED (default 0), REQUIRES (extra prerequisite task name for member 0).
set -euo pipefail

SYSTEM="$1"; DATASET="$2"; M="$3"; shift 3
QUEUE="${QUEUE:-/fs1/projects/benchmarking_PIDS/sherwin_dir/caliber_queue}"
BASE_SEED="${BASE_SEED:-0}"
mkdir -p "${QUEUE}"/{pending,running,done,failed,logs}

ARGS="$*"
AH=$(printf '%s' "${ARGS}" | sha1sum | cut -c1-8)

task_name() { echo "${SYSTEM}__${DATASET}__${AH}__seed$1"; }

write_task() {  # $1 seed, $2 requires (may be empty)
    local name; name=$(task_name "$1")
    for st in pending running done; do
        if compgen -G "${QUEUE}/${st}/${name}*" >/dev/null; then
            echo "skip ${name} (already ${st})"; return
        fi
    done
    local tmp="${QUEUE}/pending/.${name}.tmp"
    {
        echo "SYSTEM=${SYSTEM}"
        echo "DATASET=${DATASET}"
        echo "SEED=$1"
        echo "REQUIRES=$2"
        echo "ARGS=${ARGS}"
    } > "${tmp}"
    mv "${tmp}" "${QUEUE}/pending/${name}"
    echo "queued ${name}${2:+ (requires $2)}"
}

first=$(task_name "${BASE_SEED}")
write_task "${BASE_SEED}" "${REQUIRES:-}"
for (( i = 1; i < M; i++ )); do
    write_task "$(( BASE_SEED + i ))" "${first}"
done
