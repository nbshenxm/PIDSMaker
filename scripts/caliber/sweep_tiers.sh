#!/bin/bash
# Run caliber.tiers on every complete default-config ensemble (>= MIN_MEMBERS manifests),
# optionally restricted to datasets matching DS_REGEX. Reports go to artifacts_v3/caliber/reports/.
#   DS_REGEX='CADETS_E3|THEIA_E3' bash scripts/caliber/sweep_tiers.sh
set -uo pipefail
S=/fs1/projects/benchmarking_PIDS/sherwin_dir
A=$S/artifacts_v3/
MIN_MEMBERS="${MIN_MEMBERS:-5}"
DS_REGEX="${DS_REGEX:-.}"
RULES="${RULES:-single_run_static,single_run,ensemble,ensemble_upper}"
BUDGETS="${BUDGETS:-50 200}"
ls "$A/caliber/manifests" | grep '__da39a3ee5e__' | sed 's/__seed[0-9]*\.json$//' | sort | uniq -c |
while read -r n key; do
    (( n >= MIN_MEMBERS )) || continue
    system=${key%%__*}; rest=${key#*__}; dataset=${rest%%__*}
    [[ "$dataset" =~ $DS_REGEX ]] || continue
    for B in $BUDGETS; do
        python -m caliber.tiers --artifact_dir "$A" --system "$system" --dataset "$dataset" \
            --budget "$B" --rules "$RULES" --quick 2>&1 | grep -v "Warn\|read_csv\|nltk"
    done
done
