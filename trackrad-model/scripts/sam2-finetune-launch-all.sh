#!/bin/bash
set -euo pipefail

# Submits all 8 SAM2 fine-tuning ablation runs (2 checkpoints x 2 datasets x
# 2 multipliers) as independent single-node Slurm jobs via submitit. Each
# `train.py` invocation below submits its own job and returns immediately.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM2_DIR="$SCRIPT_DIR/../sam2"

CONFIGS=(
    sam2.1_hiera_b+_alldata_mult1_finetune
    sam2.1_hiera_b+_alldata_mult28_finetune
    sam2.1_hiera_b+_notest_mult1_finetune
    sam2.1_hiera_b+_notest_mult28_finetune
    sam2.1_medsam2_alldata_mult1_finetune
    sam2.1_medsam2_alldata_mult28_finetune
    sam2.1_medsam2_notest_mult1_finetune
    sam2.1_medsam2_notest_mult28_finetune
)

cd "$SAM2_DIR"
for config in "${CONFIGS[@]}"; do
    echo "Submitting $config"
    uv run --project .. python training/train.py \
        -c "configs/sam2.1_training/${config}.yaml" \
        --use-cluster 1 --num-gpus 4 --num-nodes 1 \
        --partition gen-a100.p --account m299164
done
