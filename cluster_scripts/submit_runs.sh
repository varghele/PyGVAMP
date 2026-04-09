#!/bin/bash
# ===========================================================================
# PyGVAMP — Submit multiple experiment runs
# ===========================================================================
# Submits N independent runs for a given protein, lag time, and n_states.
#
# Usage:
#   # Submit 10 runs for ab42_red at lag=10 with 10 states
#   ./submit_runs.sh --protein ab42_red \
#       --top /mnt/hdd/data/ab42/trajectories/red/topol.pdb \
#       --traj /mnt/hdd/data/ab42/trajectories/red/ \
#       --lag 10 --n_states 10 --n_runs 10
#
#   # Reversible mode
#   ./submit_runs.sh --protein ab42_red \
#       --top /mnt/hdd/data/ab42/trajectories/red/topol.pdb \
#       --traj /mnt/hdd/data/ab42/trajectories/red/ \
#       --lag 10 --n_states 10 --n_runs 10 --reversible
#
#   # Dry run (show commands without submitting)
#   ./submit_runs.sh --protein ab42_red ... --dry_run
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_RUNS=10
DRY_RUN=0
EXTRA_ARGS=()

# Collect all args, extract --n_runs and --dry_run
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_runs)   N_RUNS="$2"; shift 2;;
        --dry_run)  DRY_RUN=1; shift;;
        *)          PASS_ARGS+=("$1"); shift;;
    esac
done

echo "Submitting ${N_RUNS} runs..."
echo "Args: ${PASS_ARGS[*]}"
echo ""

for i in $(seq 0 $((N_RUNS - 1))); do
    CMD="sbatch ${SCRIPT_DIR}/run_experiment.sh ${PASS_ARGS[*]} --run ${i}"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry] $CMD"
    else
        echo -n "Run $i: "
        $CMD
    fi
done
