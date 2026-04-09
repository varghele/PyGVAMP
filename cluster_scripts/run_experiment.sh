#!/bin/bash
# ===========================================================================
# PyGVAMP — General Experiment Runner
# ===========================================================================
# Parameterized script for submitting training experiments.
# All experiment parameters are passed as arguments, not hardcoded.
#
# Usage:
#   # Single run
#   sbatch run_experiment.sh --protein ab42_red \
#       --top /mnt/hdd/data/ab42/trajectories/red/topol.pdb \
#       --traj /mnt/hdd/data/ab42/trajectories/red/ \
#       --lag 10 --n_states 10 --run 0
#
#   # Multiple runs (submit in a loop)
#   for i in $(seq 0 9); do
#     sbatch run_experiment.sh --protein ab42_red \
#       --top /path/to/topol.pdb --traj /path/to/traj/ \
#       --lag 10 --n_states 10 --run $i
#   done
#
#   # Reversible mode
#   sbatch run_experiment.sh --protein ab42_red \
#       --top /path/to/topol.pdb --traj /path/to/traj/ \
#       --lag 10 --n_states 10 --run 0 --reversible
# ===========================================================================

# ---- SLURM directives -----------------------------------------------------
#SBATCH --job-name=pygvamp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/pygv_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/pygv_%j.err

# ---- Defaults --------------------------------------------------------------
PROTEIN_NAME=""
TOPOLOGY=""
TRAJ_DIR=""
SELECTION="name CA"
LAG=""
N_STATES=""
RUN_IDX=0
PRESET="medium_schnet"
EPOCHS=50
STRIDE=1
BATCH_SIZE=2048
OUTPUT_BASE="/mnt/hdd/experiments"
REVERSIBLE=""

# ---- Parse arguments -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --protein)    PROTEIN_NAME="$2"; shift 2;;
        --top)        TOPOLOGY="$2"; shift 2;;
        --traj)       TRAJ_DIR="$2"; shift 2;;
        --selection)  SELECTION="$2"; shift 2;;
        --lag)        LAG="$2"; shift 2;;
        --n_states)   N_STATES="$2"; shift 2;;
        --run)        RUN_IDX="$2"; shift 2;;
        --preset)     PRESET="$2"; shift 2;;
        --epochs)     EPOCHS="$2"; shift 2;;
        --stride)     STRIDE="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --output)     OUTPUT_BASE="$2"; shift 2;;
        --reversible) REVERSIBLE="--reversible"; shift;;
        *)            echo "Unknown option: $1"; exit 1;;
    esac
done

# ---- Validate required args ------------------------------------------------
if [[ -z "$PROTEIN_NAME" || -z "$TOPOLOGY" || -z "$TRAJ_DIR" || -z "$LAG" || -z "$N_STATES" ]]; then
    echo "ERROR: Required arguments: --protein, --top, --traj, --lag, --n_states"
    echo "  Example:"
    echo "    sbatch $0 --protein ab42_red --top /path/topol.pdb --traj /path/traj/ --lag 10 --n_states 10"
    exit 1
fi

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

# ---- Build output path -----------------------------------------------------
MODE="std"
[[ -n "$REVERSIBLE" ]] && MODE="rev"
RUN_DIR=$(printf "%s/%s_%s/lag%s/run_%02d" "${OUTPUT_BASE}" "${PROTEIN_NAME}" "${MODE}" "${LAG}" "${RUN_IDX}")

JOB_NAME="${PROTEIN_NAME}_${MODE}_lag${LAG}_run${RUN_IDX}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Build command ---------------------------------------------------------
CMD="pygvamp"
CMD+=" --traj_dir ${TRAJ_DIR}"
CMD+=" --top ${TOPOLOGY}"
CMD+=" --lag_times ${LAG}"
CMD+=" --protein_name ${PROTEIN_NAME}"
CMD+=" --output_dir ${RUN_DIR}"
CMD+=" --preset ${PRESET}"
CMD+=" --n_states ${N_STATES}"
CMD+=" --no_discover_states"
CMD+=" --epochs ${EPOCHS}"
CMD+=" --batch_size ${BATCH_SIZE}"
CMD+=" --stride ${STRIDE}"
CMD+=" --selection '${SELECTION}'"
CMD+=" --cache"
[[ -n "$REVERSIBLE" ]] && CMD+=" ${REVERSIBLE}"

# ---- Print job info --------------------------------------------------------
echo "============================================================"
echo "PyGVAMP Experiment"
echo "============================================================"
echo "Protein:      ${PROTEIN_NAME}"
echo "Mode:         ${MODE}"
echo "Lag time:     ${LAG}"
echo "N states:     ${N_STATES}"
echo "Run:          ${RUN_IDX}"
echo "Epochs:       ${EPOCHS}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Output:       ${RUN_DIR}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Command:      ${CMD}"
echo "============================================================"

eval ${CMD}
EXIT_CODE=$?

echo "============================================================"
echo "Finished:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
