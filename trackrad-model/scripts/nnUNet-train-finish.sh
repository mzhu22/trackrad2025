#!/bin/bash
# For some reason the final validation phase of the training run doesn't always run on slurm,
# so use this script to finish things up on the head node.
PLAN="$1"
DATASET="$2"
TRAINER="$3"
DEVICE="${4:-0}"

if [[ -z "$PLAN" || -z "$DATASET" || -z "$TRAINER" ]]; then
    echo "Usage: $0 <PLAN> <DATASET> <TRAINER> <DEVICE>"
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

CUDA_VISIBLE_DEVICES="$DEVICE" nnUNetv2_train "$DATASET" 2d 0 -p "$PLAN" --c --npz -tr "$TRAINER"
CUDA_VISIBLE_DEVICES="$DEVICE" nnUNetv2_train "$DATASET" 2d 1 -p "$PLAN" --c --npz -tr "$TRAINER"
CUDA_VISIBLE_DEVICES="$DEVICE" nnUNetv2_train "$DATASET" 2d 2 -p "$PLAN" --c --npz -tr "$TRAINER"
CUDA_VISIBLE_DEVICES="$DEVICE" nnUNetv2_train "$DATASET" 2d 3 -p "$PLAN" --c --npz -tr "$TRAINER"
CUDA_VISIBLE_DEVICES="$DEVICE" nnUNetv2_train "$DATASET" 2d 4 -p "$PLAN" --c --npz -tr "$TRAINER"

echo "Job completed at $(date)"