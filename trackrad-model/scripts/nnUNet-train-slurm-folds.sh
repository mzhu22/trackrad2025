#!/bin/bash

#SBATCH --job-name=nnunet_train
#SBATCH --partition=gen-a100.p
#SBATCH --nodes=1            
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16      
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=nnUNet-train-slurm-folds.out
#SBATCH --error=nnUNet-train-slurm-folds.err

PLAN="$1"
FIRST_DATASET_ID="$2"

if [[ -z "$PLAN" || -z "$FIRST_DATASET_ID" ]]; then
    echo "Usage: $0 <PLAN> <FIRST_DATASET_ID>"
    exit 1
fi

SECOND_DATASET_ID=$((FIRST_DATASET_ID + 1))
THIRD_DATASET_ID=$((FIRST_DATASET_ID + 2))
FOURTH_DATASET_ID=$((FIRST_DATASET_ID + 3))

source /rodata/mnradonc_dev/m299164/trackrad/trackrad-model/.venv/bin/activate

# Set environment variables
export nnUNet_raw="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_results"

export nnUNet_def_n_proc=2
export nnUNet_n_proc_DA=2  # Allowed additional data augmentation processes
export OPENBLAS_NUM_THREADS=2

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run training
echo "Starting training..."
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset"$FIRST_DATASET_ID"_TrackRadFold0 2d all -p "$PLAN" --c --npz &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train Dataset"$SECOND_DATASET_ID"_TrackRadFold1 2d all -p "$PLAN" --c --npz &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train Dataset"$THIRD_DATASET_ID"_TrackRadFold2 2d all -p "$PLAN" --c --npz &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train Dataset"$FOURTH_DATASET_ID"_TrackRadFold3 2d all -p "$PLAN" --c --npz &

wait

echo "Job completed at $(date)"