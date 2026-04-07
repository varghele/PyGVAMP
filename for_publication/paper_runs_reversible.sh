#!/bin/bash
# ===========================================================================
# PyGVAMP — Paper Runs: Reversible (RevGraphVAMP)  |  SLURM Array Job
# ===========================================================================
# Runs N_RUNS independent pipeline executions per lag time for statistical
# validation. Each array task handles one (lag_time, run_index) combination.
# After all jobs finish, run paper_analysis.py to aggregate results.
#
# Usage:
#   # 5 lag times x 10 runs = 50 tasks (indices 0-49)
#   sbatch --array=0-49 paper_runs_reversible.sh
#
#   # Limit concurrency (e.g., max 5 GPU jobs at once)
#   sbatch --array=0-49%5 paper_runs_reversible.sh
# ===========================================================================

# ---- SLURM directives -----------------------------------------------------
#SBATCH --job-name=pygv_paper_rev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/paper_rev_%A_%a.out
#SBATCH --error=logs/paper_rev_%A_%a.err

# ---- USER CONFIGURATION ---------------------------------------------------

# Protein / system
PROTEIN_NAME="my_protein"
TOPOLOGY="/path/to/topology.pdb"
TRAJ_DIR="/path/to/trajectories/"
FILE_PATTERN="*.xtc"

# Atom selection (MDTraj syntax)
SELECTION="name CA"

# Lag times to sweep (one per run group)
LAG_TIMES=(5 10 20 50 100)

# Number of independent runs per lag time
N_RUNS=10

# Number of states (omit to use automatic state discovery)
# N_STATES="5"                         # uncomment to fix states manually

# Model / preset
PRESET="medium_schnet"

# Training
EPOCHS=100
STRIDE=10
BATCH_SIZE=128

# Output — structured as paper_experiments/protein/lagXns/run_YY/
OUTPUT_BASE="./paper_experiments"

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

# ---- Resolve (lag_time, run_index) from array task ID ----------------------
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    TOTAL=$((${#LAG_TIMES[@]} * N_RUNS))
    echo "ERROR: Submit as array job:  sbatch --array=0-$((TOTAL-1)) $0"
    exit 1
fi

N_LAGS=${#LAG_TIMES[@]}
LAG_IDX=$((SLURM_ARRAY_TASK_ID / N_RUNS))
RUN_IDX=$((SLURM_ARRAY_TASK_ID % N_RUNS))

if [ ${LAG_IDX} -ge ${N_LAGS} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} exceeds ${N_LAGS} lag times x ${N_RUNS} runs"
    exit 1
fi

LAG=${LAG_TIMES[$LAG_IDX]}
RUN_DIR=$(printf "%s/%s_rev/lag%sns/run_%02d" "${OUTPUT_BASE}" "${PROTEIN_NAME}" "${LAG}" "${RUN_IDX}")

# Update job name
JOB_NAME="${PROTEIN_NAME}_rev_lag${LAG}_run${RUN_IDX}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Build command ---------------------------------------------------------
CMD="python run_pipeline.py"
CMD+=" --traj_dir ${TRAJ_DIR}"
CMD+=" --top ${TOPOLOGY}"
CMD+=" --lag_times ${LAG}"
CMD+=" --protein_name ${PROTEIN_NAME}"
CMD+=" --output_dir ${RUN_DIR}"
CMD+=" --reversible"

if [ -n "${PRESET}" ]; then
    CMD+=" --preset ${PRESET}"
fi

if [ -n "${N_STATES}" ]; then
    CMD+=" --n_states ${N_STATES}"
    CMD+=" --no_discover_states"
fi

CMD+=" --epochs ${EPOCHS}"
CMD+=" --batch_size ${BATCH_SIZE}"
CMD+=" --stride ${STRIDE}"
CMD+=" --selection '${SELECTION}'"
CMD+=" --cache"

# ---- Print job info --------------------------------------------------------
echo "============================================================"
echo "PyGVAMP Paper Run — Reversible (RevGraphVAMP)"
echo "============================================================"
echo "Job:          ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID})"
echo "Protein:      ${PROTEIN_NAME}"
echo "Lag time:     ${LAG} ns  (index ${LAG_IDX}/${N_LAGS})"
echo "Run:          ${RUN_IDX}/${N_RUNS}"
echo "Output:       ${RUN_DIR}"
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
