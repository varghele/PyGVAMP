#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin strict reproduction (Ghorbani et al. 2022, Table S1)
# ===========================================================================
# Target: VAMP-2 = 3.78 ± 0.02 on the DE Shaw 2F4K-0 trajectory.
# All hyperparameters are explicit on the pygvamp command line below — there
# is no per-experiment preset class.  See claude/VILLIN_REPRO_LOG.md for the
# protocol provenance and gotchas.
#
# Submit as a 10-seed array (throttled to one GPU at a time):
#   sbatch --array=0-9%1 cluster_scripts/villin_repro.sh
#
# Each task: SLURM_ARRAY_TASK_ID -> --seed 0..9.  Seeds drive the train/val
# random_split and the dataset RNG, so each run gets a distinct split.
# Seed values are persisted in pipeline_summary.json under config.seed.
#
# Timestep gotcha: DE Shaw DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-9%1 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro/seed_%02d" "${SEED}")

JOB_NAME="villin_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin strict reproduction (array task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table S1)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# Every paper-specified value is right here.  No hidden state in a preset.
# Categories below match Ghorbani 2022 protocol:
#   Data        : 2F4K-0 c-alpha trajectory, 7 DCDs, 200 ps/frame, 35 Cα atoms
#   Architecture: SchNet, 4 graph layers × 16 neurons, 16 Gaussians, no attention
#   Training    : Adam-style, lr=5e-4, batch 1000, 100 epochs, 70/30 split, k=4
#   Lag         : 20 ns
#   Discovery   : OFF (--no_discover_states + max_retrains=0)
#   Auto-stride : OFF (single lag time)
#   Reversible  : OFF (standard VAMP-2)

pygvamp \
    --traj_dir /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/2F4K-0-c-alpha/ \
    --top      /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/topol.pdb \
    --file_pattern '2F4K-0-c-alpha-*.dcd' \
    --protein_name villin \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        schnet \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     4 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --hidden_dim            16 \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           10 \
    --gaussian_expansion_dim 16 \
    --no_use_attention \
    --lr           5e-4 \
    --weight_decay 1e-5 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --early_stopping_patience  8 \
    --early_stopping_tol       5e-4 \
    --early_stopping_min_epochs 20 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
