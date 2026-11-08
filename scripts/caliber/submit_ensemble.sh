#!/bin/bash
# Submit an M-member seed ensemble for one (system, dataset).
#
# Member 0 runs alone first: it builds the shared upstream artifacts
# (construction ... batching), whose paths do not depend on the training seed.
# Members 1..M-1 then run afterwards (afterok), split into G jobs that each
# train a chunk of seeds sequentially. Chunking keeps the number of submitted
# jobs small (the account has a MaxSubmitJobs limit) and avoids racing on the
# shared task directories.
#
#   scripts/caliber/submit_ensemble.sh SYSTEM DATASET M [extra pidsmaker args...]
#   e.g. scripts/caliber/submit_ensemble.sh velox CADETS_E3 10
#        scripts/caliber/submit_ensemble.sh orthrus THEIA_E3 10 --training.stable_optim=True
#
# Env: BASE_SEED (default 0), NGROUPS (parallel jobs for members 1..M-1, default 3),
#      SBATCH_EXTRA (e.g. "--time=24:00:00"), AFTER (job id member 0 must wait for),
#      FIRST_JOB (reuse an already-submitted member-0 job instead of submitting one).
set -euo pipefail

SYSTEM="$1"; DATASET="$2"; M="$3"; shift 3
BASE_SEED="${BASE_SEED:-0}"
NGROUPS="${NGROUPS:-3}"
TAG="${SYSTEM}-${DATASET}"
mkdir -p logs/slurm

DEP0=()
[[ -n "${AFTER:-}" ]] && DEP0=(--dependency="afterok:${AFTER}")

if [[ -n "${FIRST_JOB:-}" ]]; then
    first="${FIRST_JOB}"
    echo "member 0: reusing job ${first}"
else
    first=$(sbatch --parsable --job-name="${TAG}" ${SBATCH_EXTRA:-} "${DEP0[@]}" \
        --export=ALL,SEEDS="${BASE_SEED}" \
        scripts/caliber/run_member.sbatch "${SYSTEM}" "${DATASET}" "$@")
    echo "member 0 (seed ${BASE_SEED}): job ${first}"
fi

(( M > 1 )) || exit 0
for (( g = 0; g < NGROUPS; g++ )); do
    seeds=""
    for (( i = 1 + g; i < M; i += NGROUPS )); do seeds+="$(( BASE_SEED + i )) "; done
    [[ -n "${seeds}" ]] || continue
    jid=$(sbatch --parsable --job-name="${TAG}" ${SBATCH_EXTRA:-} \
        --dependency="afterok:${first}" \
        --export=ALL,SEEDS="${seeds% }" \
        scripts/caliber/run_member.sbatch "${SYSTEM}" "${DATASET}" "$@")
    echo "seeds [${seeds% }]: job ${jid} (after ${first})"
done
