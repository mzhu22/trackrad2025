#!/bin/bash

DATASET="$1"  # e.g., 491
CONFIG="$2"  # e.g., 2d
PLAN="$3"  # e.g., nnUNetPlans
TRAINER="$4"  # e.g., nnUNetTrainerDAMike

if [ -z "$DATASET" ] || [ -z "$CONFIG" ] || [ -z "$PLAN" ] || [ -z "$TRAINER" ]; then
    echo "Usage: $0 <dataset> <config> <plan> <trainer>"
    exit 1
fi

# Folds (0, 1, 2, 3) on one node
sbatch _nnUNet-train-slurm-1.sh "$DATASET" "$CONFIG" "$PLAN" "$TRAINER"
# Folds (4, all) on another node
sbatch _nnUNet-train-slurm-2.sh "$DATASET" "$CONFIG" "$PLAN" "$TRAINER"