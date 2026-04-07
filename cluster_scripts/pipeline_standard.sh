#!/bin/bash
# ===========================================================================
# PyGVAMP — Standard (VAMP-2) Full Pipeline  |  SLURM Array Job
# ===========================================================================
# Each array task runs the complete pipeline (preparation → training →
# analysis) for ONE lag time.  Preparation caching means the dataset is
# built only once (first task to reach it); subsequent tasks reuse it.
#
# Usage:
#   sbatch --array=0-4 pipeline_standard.sh        # 5 lag times
#   sbatch --array=0-4%2 pipeline_standard.sh      # max 2 concurrent
# ===========================================================================

# ---- SLURM directives -----------------------------------------------------
#SBATCH --job-name=pygv_std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/std_%A_%a.out
#SBATCH --error=logs/std_%A_%a.err

# ---- USER CONFIGURATION — edit this section --------------------------------

# Protein / system
PROTEIN_NAME="my_protein"
TOPOLOGY="/path/to/topology.pdb"
TRAJ_DIR="/path/to/trajectories/"
FILE_PATTERN="*.xtc"

# Atom selection (MDTraj syntax)
SELECTION="name CA"

# Lag times — one per array task (index into this array with SLURM_ARRAY_TASK_ID)
LAG_TIMES=(5 10 20 50 100)

# Number of states (omit to use automatic state discovery)
# N_STATES="5 8"                       # uncomment to fix states manually
# MIN_STATES=2                         # only used with state discovery
# MAX_STATES=10                        # only used with state discovery

# Model / preset (pick ONE)
PRESET="medium_schnet"                  # preset name — see `python run_pipeline.py --help`
# MODEL="schnet"                       # or pick just the encoder type

# Training
EPOCHS=100
STRIDE=10
BATCH_SIZE=128

# Output
OUTPUT_DIR="./experiments"
CACHE="--cache"                         # set to "" to disable caching

# Conda
CONDA_ENV="PyGVAMP5"

# ---- END USER CONFIGURATION -----------------------------------------------

# ---- Environment setup -----------------------------------------------------
module purge
module load CUDA/12.4.0
module load Anaconda3/2024.02-1

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

mkdir -p logs

# ---- Resolve lag time for this array task ----------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: This script must be submitted as an array job."
    echo "  sbatch --array=0-$((${#LAG_TIMES[@]}-1)) $0"
    exit 1
fi

LAG=${LAG_TIMES[$SLURM_ARRAY_TASK_ID]}
if [ -z "${LAG}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range (${#LAG_TIMES[@]} lag times defined)"
    exit 1
fi

# Update job name to include lag time
JOB_NAME="${PROTEIN_NAME}_std_lag${LAG}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Build command ---------------------------------------------------------
CMD="python run_pipeline.py"
CMD+=" --traj_dir ${TRAJ_DIR}"
CMD+=" --top ${TOPOLOGY}"
CMD+=" --lag_times ${LAG}"
CMD+=" --protein_name ${PROTEIN_NAME}"
CMD+=" --output_dir ${OUTPUT_DIR}"

# Preset or model
if [ -n "${PRESET}" ]; then
    CMD+=" --preset ${PRESET}"
elif [ -n "${MODEL}" ]; then
    CMD+=" --model ${MODEL}"
fi

# States
if [ -n "${N_STATES}" ]; then
    CMD+=" --n_states ${N_STATES}"
    CMD+=" --no_discover_states"
else
    [ -n "${MIN_STATES}" ] && CMD+=" --min_states ${MIN_STATES}"
    [ -n "${MAX_STATES}" ] && CMD+=" --max_states ${MAX_STATES}"
fi

# Training overrides
CMD+=" --epochs ${EPOCHS}"
CMD+=" --batch_size ${BATCH_SIZE}"
CMD+=" --stride ${STRIDE}"
CMD+=" --selection '${SELECTION}'"
[ -n "${FILE_PATTERN}" ] && true  # file_pattern is set via config/preset
[ -n "${CACHE}" ] && CMD+=" ${CACHE}"

# ---- Print job info --------------------------------------------------------
echo "============================================================"
echo "PyGVAMP Standard Pipeline"
echo "============================================================"
echo "Job:          ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID})"
echo "Protein:      ${PROTEIN_NAME}"
echo "Lag time:     ${LAG} ns"
echo "Preset:       ${PRESET:-custom}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Command:      ${CMD}"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
eval ${CMD}
EXIT_CODE=$?

echo "============================================================"
echo "Finished:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
