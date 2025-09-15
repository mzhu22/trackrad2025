#!/bin/bash
DATASET_ID="$1"
PLAN="$2"
TRAINER="$3"

if [[ -z "$DATASET_ID" || -z "$PLAN" || -z "$TRAINER" ]]; then
    echo "Usage: $0 <DATASET_ID> <PLAN> <TRAINER>"
    exit 1
fi

export nnUNet_raw="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_results"

export nnUNet_def_n_proc=2
export OPENBLAS_NUM_THREADS=2

nnUNetv2_find_best_configuration "$DATASET_ID" -c 2d -p "$PLAN" -np 2 -tr "$TRAINER"