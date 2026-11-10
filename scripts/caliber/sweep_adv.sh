#!/bin/bash
# Detection under realizable attacks: caliber.tiers on the clean weight-saving ensemble
# (CLEAN group) and on every adversarial group scored with the same frozen weights.
#   bash scripts/caliber/sweep_adv.sh
set -uo pipefail
S=/fs1/projects/benchmarking_PIDS/sherwin_dir
A=$S/artifacts_v3/
CLEAN="${CLEAN:-488377d761}"
RULES="${RULES:-single_run_static,single_run,ensemble,ensemble_upper}"
BUDGETS="${BUDGETS:-50 200}"
for pair in "velox CADETS_E3" "velox THEIA_E3" "orthrus CADETS_E3" "orthrus THEIA_E3"; do
    set -- $pair
    groups=$(grep -l -- '"--adversary"' "$A"/caliber/manifests/"$1__$2"__*.json 2>/dev/null |
             xargs -r -n1 basename | sed 's/__seed[0-9]*\.json$//; s/.*__//' | sort -u)
    for g in "$CLEAN" $groups; do
        spec=$(python -c "import json,glob; m=json.load(open(sorted(glob.glob('$A/caliber/manifests/$1__$2__${g}__seed*.json'))[0])); a=m['argv']; print(a[a.index('--adversary')+1] if '--adversary' in a else 'clean')")
        for B in $BUDGETS; do
            echo "#### $1 $2 group=$g attack=$spec B=$B"
            python -m caliber.tiers --artifact_dir "$A" --system "$1" --dataset "$2" --group "$g" \
                --budget "$B" --rules "$RULES" --quick --single_all 2>&1 | grep -v "Warn\|read_csv\|nltk"
        done
    done
done
