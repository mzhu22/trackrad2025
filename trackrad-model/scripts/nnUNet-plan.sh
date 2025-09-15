#!/bin/bash
DATASET_ID="$1"
if [ $# -lt 1 ]; then
    echo "Usage: $0 <DATASET_ID>"
    exit 1
fi

export nnUNet_raw="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/rodata/mnradonc_dev/m299164/trackrad/datasets/nnUNet/nnUNet_results"

export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1
export nnUNet_def_n_proc=2
export nnUNet_n_proc_DA=2  # Allowed additional data augmentation processes
export OPENBLAS_NUM_THREADS=2

nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -npfp 2 -np 2
# nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -pl nnUNetPlannerResEncM
# nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -pl nnUNetPlannerResEncL

# nnUNetv2_find_best_configuration 451 -p "$PLAN"