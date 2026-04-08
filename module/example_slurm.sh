#!/bin/bash
#SBATCH --job-name=pygvamp_${SLURM_ARRAY_TASK_ID}states
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/pygvamp_%A_%a.out
#SBATCH --error=logs/pygvamp_%A_%a.err

# ============================================================================
# Example SLURM submission script using the PyGVAMP module.
#
# Usage:
#   # Single run with 5 states
#   sbatch example_slurm.sh
#
#   # Array job sweeping n_states 3-10
#   sbatch --array=3-10 example_slurm.sh
# ============================================================================

# Load modules — no conda activate needed
module purge
module load 12.8
module load pygvamp/1.0.0

mkdir -p logs

echo "Job ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID:-none}) started at $(date)"
echo "PyGVAMP version: $(pygvamp --version 2>/dev/null || python -c 'import pygv; print(pygv.__version__)')"
nvidia-smi

# ── Configure your run ────────────────────────────────────────────────
# Adjust these to match your system:

PROTEIN="my_protein"
TOPOLOGY="/path/to/topology.pdb"
TRAJ_DIR="/path/to/trajectories/"
OUTPUT_DIR="/path/to/output"

# Use array task ID for n_states if running as array job, otherwise default
N_STATES=${SLURM_ARRAY_TASK_ID:-5}

# ── Run the pipeline ──────────────────────────────────────────────────

pygvamp \
    --protein_name "$PROTEIN" \
    --top "$TOPOLOGY" \
    --traj_dir "$TRAJ_DIR" \
    --file_pattern "*.xtc" \
    --selection "name CA" \
    --preset medium_schnet \
    --n_states "$N_STATES" \
    --lag_times 10 20 50 \
    --stride 10 \
    --epochs 100 \
    --batch_size 128 \
    --use_cache \
    --output_dir "$OUTPUT_DIR" \
    --run_name "${PROTEIN}_${N_STATES}states"

echo "Job completed at $(date)"
