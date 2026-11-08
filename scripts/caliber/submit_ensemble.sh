#!/bin/bash
# Submit an M-member seed ensemble for one (system, dataset).
#
# Member 0 runs alone first: it builds the shared upstream artifacts
# (construction ... batching), whose paths do not depend on the training seed.
# Members 1..M-1 then start in parallel (afterok) and reuse those artifacts,
# so they never race on the same task directory.
#
#   scripts/caliber/submit_ensemble.sh SYSTEM DATASET M [extra pidsmaker args...]
#   e.g. scripts/caliber/submit_ensemble.sh velox CADETS_E3 10
#        scripts/caliber/submit_ensemble.sh orthrus THEIA_E3 10 --training.stable_optim=True
#
# Env: BASE_SEED (default 0), MAX_PARALLEL (default 5), SBATCH_EXTRA (e.g. "--time=24:00:00").
set -euo pipefail

SYSTEM="$1"; DATASET="$2"; M="$3"; shift 3
BASE_SEED="${BASE_SEED:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
TAG="${SYSTEM}-${DATASET}"
mkdir -p logs/slurm

first=$(BASE_SEED="${BASE_SEED}" sbatch --parsable --job-name="${TAG}" ${SBATCH_EXTRA:-} \
    --export=ALL,BASE_SEED="${BASE_SEED}" \
    scripts/caliber/run_member.sbatch "${SYSTEM}" "${DATASET}" "$@")
echo "member 0 (seed ${BASE_SEED}): job ${first}"

if (( M > 1 )); then
    rest=$(sbatch --parsable --job-name="${TAG}" ${SBATCH_EXTRA:-} \
        --array="1-$((M - 1))%${MAX_PARALLEL}" --dependency="afterok:${first}" \
        --export=ALL,BASE_SEED="${BASE_SEED}" \
        scripts/caliber/run_member.sbatch "${SYSTEM}" "${DATASET}" "$@")
    echo "members 1..$((M - 1)): array job ${rest} (after ${first})"
fi
