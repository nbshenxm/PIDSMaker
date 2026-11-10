#!/bin/bash
# Deployment study for one or more SYSTEM:DATASET pairs (CPU only).
#   PAIRS="velox:CADETS_E3 orthrus:THEIA_E3" DETECTOR=model sbatch scripts/caliber/deploy_study.sbatch
# DETECTOR=allowlist also exports the training-period executables first (needs the DB).
#SBATCH --job-name=cal-deploy
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=/fs1/projects/benchmarking_PIDS/sherwin_dir/caliber_queue/logs/%x-%j.out
set -o pipefail
S=/fs1/projects/benchmarking_PIDS/sherwin_dir
A=$S/artifacts_v3/
cd $S/PIDSMaker3
source ~/.bashrc >/dev/null 2>&1
module load miniforge3 >/dev/null 2>&1
conda activate pids3
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$PWD
set -u
DETECTOR="${DETECTOR:-model}"
for pair in $PAIRS; do
    system=${pair%%:*}; dataset=${pair#*:}
    if [[ "$DETECTOR" == allowlist && ! -f "$A/caliber/labels/${dataset}__train_procs.csv.gz" ]]; then
        read -r DB_HOST DB_PORT _ < $S/db_server.info
        python -m caliber.pidsmaker_bridge train_procs "$system" "$dataset" --artifact_dir "$A" \
            --database_host "$DB_HOST" --database_port "$DB_PORT" || continue
    fi
    python -m caliber.deploy_study --artifact_dir "$A" --system "$system" --dataset "$dataset" \
        --detector "$DETECTOR" 2>&1 | grep -v "Warn\|read_csv\|nltk"
done
