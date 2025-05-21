#!/bin/bash
#SBATCH --job-name=ab42_train_nc${SLURM_ARRAY_TASK_ID}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# Set job name properly with array ID since the above won't expand at submission time
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    JOB_NAME="atr_train_nc${SLURM_ARRAY_TASK_ID}"
    scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME}
    echo "Updated job name to: ${JOB_NAME}"
fi

# Load required modules
module purge
module load CUDA/12.4.0
module load Anaconda3/2024.02-1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate PyGVAMP5

# Create log directory if it doesn't exist
mkdir -p logs

# Print job array information
echo "Running job array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Using num_classes = ${SLURM_ARRAY_TASK_ID}"

# Print GPU information
nvidia-smi

# Run the training script with the array task ID as num_classes
python run_training.py \
    --protein_name "ATR" \
    --top "XXXXXXXXXXXXXXXX" \
    --traj_dir "XXXXXXXXXXXXXXXX" \
    --file_pattern "*.xtc" \
    --selection "name CA" \
    --val_split 0.05 \
    --sample_validate_every 100 \
    --stride 10 \
    --lag_time 20 \
    --n_neighbors 20 \
    --node_embedding_dim 32 \
    --gaussian_expansion_dim 16 \
    --node_dim 32 \
    --edge_dim 16 \
    --hidden_dim 32 \
    --output_dim 32 \
    --n_interactions 4 \
    --activation 'tanh' \
    --use_attention True \
    --n_states ${SLURM_ARRAY_TASK_ID} \
    --clf_hidden_dim 32 \
    --clf_num_layers 2 \
    --clf_dropout 0.01 \
    --clf_activation 'leaky_relu' \
    --clf_norm 'LayerNorm' \
    --use_embedding True \
    --embedding_in_dim 42 \
    --embedding_hidden_dim 64 \
    --embedding_out_dim 32 \
    --embedding_num_layers 2 \
    --embedding_dropout 0.01 \
    --embedding_act 'leaky_relu' \
    --embedding_norm None \
    --epochs 25 \
    --batch_size 256 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --clip_grad None \
    --cpu False \
    --max_tau 200 \
    --output_dir 'area57' \
    --cache_dir 'area57/cache' \
    --use_cache True \
    --save_every 0 \
    --run_name 'ab42' \

# Print completion message
echo "Job array task ${SLURM_ARRAY_TASK_ID} completed at: $(date)"
