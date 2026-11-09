#!/bin/bash
# Scale-out experiment plan (2026-11-09). Enqueues tasks for the worker pool.
# Order matters (the queue is FIFO): breadth over systems first, then
# leave-one-day-out (LODO) ensembles, then new datasets.
#
# Per dataset, member 0 of each system waits for member 0 of the previous system
# (REQUIRES chain) so shared upstream artifacts (construction, transformation, ...)
# are never built by two tasks at once.
#
#   bash scripts/caliber/plan_scaleout.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
EQ=scripts/caliber/enqueue.sh
AH_EMPTY=$(printf '' | sha1sum | cut -c1-8)

chain() {  # chain DATASET M "SYS1 SYS2 ..." [extra args...]
    local ds="$1" m="$2" systems="$3"; shift 3
    local args="$*" ah prev=""
    ah=$(printf '%s' "${args}" | sha1sum | cut -c1-8)
    for s in ${systems}; do
        REQUIRES="${prev}" bash "${EQ}" "${s}" "${ds}" "${m}" "$@" | tail -1
        prev="${s}__${ds}__${ah}__seed0"
    done
}

ALL="velox orthrus kairos magic flash nodlink rcaid threatrace ocrapt"
NEW="kairos magic flash nodlink rcaid threatrace ocrapt"

# P1: breadth. All systems on the E3 datasets, 5 seeds (velox/orthrus already have
# 10 seeds on CADETS/THEIA/CLEARSCOPE; enqueue skips tasks already done).
for ds in CADETS_E3 THEIA_E3 CLEARSCOPE_E3; do chain "${ds}" 5 "velox ${NEW}"; done
for ds in TRACE_E3 FIVEDIRECTIONS_E3; do chain "${ds}" 5 "${ALL}"; done

# P2: leave-one-day-out ensembles (velox, orthrus), 2 seeds per held-out training day.
for ds in CADETS_E3 THEIA_E3; do
    days=$(python - "${ds}" <<'EOF'
import sys
from pidsmaker.config.config import DATASET_DEFAULT_CONFIG as C
print(" ".join(C[sys.argv[1]]["train_dates"]))
EOF
)
    for d in ${days}; do chain "${ds}" 2 "velox orthrus" --holdout_train_day "${d}"; done
done

# P3: new datasets (multi-host OpTC/E5, long-horizon ATLASv2/CARBANAKv2), 10 seeds.
for ds in THEIA_E5 CLEARSCOPE_E5 optc_h051 optc_h201 optc_h501 ATLASV2_EDR CARBANAKV2_EDR; do
    chain "${ds}" 10 "velox orthrus"
done
