#!/bin/bash
# ===========================================================================
# PyGVAMP — State Discovery (General)
# ===========================================================================
# Runs preparation + Graph2Vec + clustering to determine n_states.
# Submit once per protein system, then use the result in experiment runs.
#
# Usage:
#   sbatch run_discovery.sh --protein ab42_red \
#       --top /mnt/hdd/data/ab42/trajectories/red/topol.pdb \
#       --traj /mnt/hdd/data/ab42/trajectories/red/
#
#   # Custom selection / stride
#   sbatch run_discovery.sh --protein ab42_red \
#       --top /path/topol.pdb --traj /path/traj/ \
#       --selection "name CA" --stride 1
#
#   # Check result:
#   grep "Recommended n_states" logs/disc_*.out
# ===========================================================================

#SBATCH --job-name=pygv_disc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/disc_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/disc_%j.err

# ---- Defaults --------------------------------------------------------------
PROTEIN_NAME=""
TOPOLOGY=""
TRAJ_DIR=""
SELECTION="name CA"
STRIDE=1
PRESET="medium_schnet"
OUTPUT_BASE="/mnt/hdd/experiments"

# ---- Parse arguments -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --protein)    PROTEIN_NAME="$2"; shift 2;;
        --top)        TOPOLOGY="$2"; shift 2;;
        --traj)       TRAJ_DIR="$2"; shift 2;;
        --selection)  SELECTION="$2"; shift 2;;
        --stride)     STRIDE="$2"; shift 2;;
        --preset)     PRESET="$2"; shift 2;;
        --output)     OUTPUT_BASE="$2"; shift 2;;
        *)            echo "Unknown option: $1"; exit 1;;
    esac
done

# ---- Validate required args ------------------------------------------------
if [[ -z "$PROTEIN_NAME" || -z "$TOPOLOGY" || -z "$TRAJ_DIR" ]]; then
    echo "ERROR: Required arguments: --protein, --top, --traj"
    echo "  Example:"
    echo "    sbatch $0 --protein ab42_red --top /path/topol.pdb --traj /path/traj/"
    exit 1
fi

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

OUTPUT_DIR="${OUTPUT_BASE}/${PROTEIN_NAME}/discovery"
JOB_NAME="${PROTEIN_NAME}_disc"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Run preparation only (skip training) ----------------------------------
echo "============================================================"
echo "PyGVAMP — State Discovery"
echo "============================================================"
echo "Protein:      ${PROTEIN_NAME}"
echo "Topology:     ${TOPOLOGY}"
echo "Trajectories: ${TRAJ_DIR}"
echo "Selection:    ${SELECTION}"
echo "Stride:       ${STRIDE}"
echo "Output:       ${OUTPUT_DIR}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "============================================================"

pygvamp \
    --traj_dir "${TRAJ_DIR}" \
    --top "${TOPOLOGY}" \
    --lag_times 1 \
    --protein_name "${PROTEIN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --preset "${PRESET}" \
    --stride "${STRIDE}" \
    --selection "${SELECTION}" \
    --cache \
    --skip_training

EXIT_CODE=$?

echo "============================================================"
echo "Finished:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo ""
echo "Next step: check the recommended n_states, then submit experiments:"
echo "  grep 'Recommended n_states' /mnt/hdd/experiments/logs/disc_${SLURM_JOB_ID}.out"
echo ""
echo "  ./cluster_scripts/submit_runs.sh \\"
echo "      --protein ${PROTEIN_NAME} --top ${TOPOLOGY} --traj ${TRAJ_DIR} \\"
echo "      --lag <LAG> --n_states <N> --n_runs 10"
echo "============================================================"

exit ${EXIT_CODE}
