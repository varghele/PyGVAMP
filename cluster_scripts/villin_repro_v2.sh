#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v2 (closer-to-paper architecture)
# ===========================================================================
# Same protocol as villin_repro.sh, with three deviations from v1 reverted
# toward Ghorbani et al. 2022:
#
#   1. No pre-encoder embedding MLP (--no_use_embedding).
#      v1 wrapped node features in a 35→64→32 ReLU MLP; the paper feeds
#      one-hot atom-type identity straight into SchNet's first linear, which
#      is functionally equivalent to nn.Embedding(35, hidden_dim).
#
#   2. Classifier head stripped of dropout + BatchNorm, reduced to a single
#      linear → softmax (--clf_num_layers 1, --clf_dropout 0, --clf_norm none).
#      v1 had a 2-layer 64-hidden MLP with dropout=0.1 and BN; the paper
#      describes only a softmax output.
#
#   3. No early stopping.  All --early_stopping_* flags omitted so training
#      runs the full 100 epochs (matches paper).  In v1, 4/10 seeds bailed
#      at the early_stopping_min_epochs floor of 20; this v2 lets every seed
#      see the full schedule.
#
# Everything else (data, k=4, lag=20 ns, n_neighbors=10, hidden_dim=16,
# n_interactions=4, gaussian_expansion_dim=16, batch=1000, lr=5e-4,
# val_split=0.3, weight_decay=1e-5, no attention, no discovery) matches v1.
#
# Submit as a 10-seed array (throttled to one GPU at a time):
#   sbatch --array=0-9%1 cluster_scripts/villin_repro_v2.sh
#
# Timestep gotcha: DE Shaw DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v2_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v2_%A_%a.err

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
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v2/seed_%02d" "${SEED}")

JOB_NAME="villin_v2_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v2 (array task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table S1)"
echo "Diffs vs v1: no pre-encoder MLP, linear softmax head, no early stopping"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
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
    --no_use_embedding \
    --clf_num_layers 1 \
    --clf_dropout    0 \
    --clf_norm       none \
    --lr           5e-4 \
    --weight_decay 1e-5 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
