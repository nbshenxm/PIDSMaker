#!/bin/bash
# Restore one or more PIDSMaker .dump files into the SHARED postgres server
# (scripts/caliber/db_server.sbatch), using the pg tools inside postgres.sif.
# Skips databases that already contain tables.
#
#   scripts/caliber/load_dump.sh /path/to/atlasv2_edr.dump [/path/to/carbanakv2_edr.dump ...]
#
# Download the dumps first (needs a Google OAuth access token, see
# docs/docs/ten-minute-install.md), e.g. from the old checkout's data/ dir:
#   ./download_datasets.sh atlasv2_edr,carbanakv2_edr <ACCESS_TOKEN>
set -euo pipefail

DB_INFO="${DB_INFO:-/fs1/projects/benchmarking_PIDS/sherwin_dir/db_server.info}"
SIF="${SIF:-/fs1/projects/benchmarking_PIDS/sherwin_dir/PIDSMaker/scripts/apptainer/postgres.sif}"
read -r DB_HOST DB_PORT _ < "${DB_INFO}"
module load apptainer >/dev/null 2>&1 || true
C="$(command -v apptainer || command -v singularity)"
export PGPASSWORD=postgres

pg() { "${C}" exec --bind /fs1 "${SIF}" "$@"; }

for dump in "$@"; do
    dump="$(readlink -f "${dump}")"
    db="$(basename "${dump}" .dump)"
    echo "== ${dump} -> ${db} @ ${DB_HOST}:${DB_PORT}"
    if pg psql -h "${DB_HOST}" -p "${DB_PORT}" -U postgres -lqt | cut -d'|' -f1 | grep -qw "${db}"; then
        n=$(pg psql -h "${DB_HOST}" -p "${DB_PORT}" -U postgres -d "${db}" -tAc \
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';")
        if [[ "${n}" -gt 0 ]]; then echo "   already has ${n} tables, skipping"; continue; fi
    else
        pg psql -h "${DB_HOST}" -p "${DB_PORT}" -U postgres -c "CREATE DATABASE \"${db}\";"
    fi
    pg pg_restore -h "${DB_HOST}" -p "${DB_PORT}" -U postgres --no-owner --no-privileges \
        -j 4 -d "${db}" "${dump}" || echo "   pg_restore reported warnings (often benign)"
    n=$(pg psql -h "${DB_HOST}" -p "${DB_PORT}" -U postgres -d "${db}" -tAc \
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';")
    echo "   ${db}: ${n} tables"
done
