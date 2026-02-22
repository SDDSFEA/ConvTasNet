#!/bin/bash
set -euo pipefail

EXCLUDE_FILE="/lustre/teams/mmai/mmai-job-setting/exclude_nodes.txt"

EXCLUDE_NODES=""
if [[ -f "$EXCLUDE_FILE" ]]; then
  EXCLUDE_NODES=$(grep -v '^\s*#' "$EXCLUDE_FILE" | grep -v '^\s*$' | paste -sd,)
fi

echo "[INFO] Submitting with --exclude=${EXCLUDE_NODES}"
# sbatch --exclude="$EXCLUDE_NODES" run_film.sh
# sbatch --exclude="$EXCLUDE_NODES" run_att2.sh
# sbatch --exclude="$EXCLUDE_NODES"  run_att1_32.sh

# sbatch --exclude="$EXCLUDE_NODES" run_att1_clean360_256.sh
# sbatch --exclude="$EXCLUDE_NODES" run_att1_clean360_128.sh
# sbatch --exclude="$EXCLUDE_NODES" run_att1_clean360_64.sh
# sbatch --exclude="$EXCLUDE_NODES" run_att1_clean360_32.sh

sbatch --exclude="$EXCLUDE_NODES" run_att1_noise360_256.sh
sbatch --exclude="$EXCLUDE_NODES" run_att1_noise360_128.sh
sbatch --exclude="$EXCLUDE_NODES" run_att1_noise360_64.sh
sbatch --exclude="$EXCLUDE_NODES" run_att1_noise360_32.sh
