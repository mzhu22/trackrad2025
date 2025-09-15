#!/bin/bash

#SBATCH --job-name=nnunet_train
#SBATCH --partition=gen-a100.p
#SBATCH --nodes=1            
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16      
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=nnUNet-train-slurm-2.out
#SBATCH --error=nnUNet-train-slurm-2.err

DATASET="$1"
CONFIG="$2"
PLAN="$3"
TRAINER="$4"

if [ -z "$DATASET" ] || [ -z "$CONFIG" ] || [ -z "$PLAN" ] || [ -z "$TRAINER" ]; then
    echo "Usage: $0 <dataset> <config> <plan> <trainer>"
    exit 1
fi


source /rodata/mnradonc_dev/m299164/trackrad/trackrad-model/.venv/bin/activate

# Set environment variables
export nnUNet_raw="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_results"

export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1
export nnUNet_def_n_proc=2
export nnUNet_n_proc_DA=2  # Allowed additional data augmentation processes
export OPENBLAS_NUM_THREADS=2

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run training
echo "Starting training..."
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train "$DATASET" "$CONFIG" 4 -p "$PLAN" --c --npz -tr "$TRAINER" &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train "$DATASET" "$CONFIG" all -p "$PLAN" --c --npz -tr "$TRAINER" &

wait

echo "Job completed at $(date)"