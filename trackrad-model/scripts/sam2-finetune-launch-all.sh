#!/bin/bash
set -euo pipefail

# Runs all 15 SAM2 fine-tuning runs (5 starting models x manual/semiauto/combined
# training data) sequentially on this machine, using the configs in
# sam2/sam2/configs/sam2.1_training/. Prerequisites, from trackrad-model/:
#   0. uv run python scripts/download_data.py                (data); bash resources/download_*.sh (checkpoints)
#   1. uv run python propagate_labels.py                     (semi-auto data)
#   2. uv run python scripts/prepare_sam2_finetune_data.py   (manual data + file lists)
# Override the GPU count with NUM_GPUS (default 2).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM2_DIR="$SCRIPT_DIR/../sam2"
NUM_GPUS="${NUM_GPUS:-2}"

MODELS=(sam2.1_hiera_t sam2.1_hiera_s sam2.1_hiera_b+ sam2.1_hiera_l sam2.1_medsam2)
DATASETS=(manual semiauto combined)

cd "$SAM2_DIR"
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        config="${model}_${dataset}_finetune"
        echo "Training $config"
        uv run --project .. python training/train.py \
            -c "configs/sam2.1_training/${config}.yaml" \
            --use-cluster 0 --num-gpus "$NUM_GPUS"
    done
done
