#!/bin/bash
# Downloads the four SAM 2.1 checkpoints into this directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
CHECKPOINTS=(
    sam2.1_hiera_tiny.pt
    sam2.1_hiera_small.pt
    sam2.1_hiera_base_plus.pt
    sam2.1_hiera_large.pt
)

if command -v curl &> /dev/null; then
    download() { curl -fL -o "$1" "$2"; }
elif command -v wget &> /dev/null; then
    download() { wget -O "$1" "$2"; }
else
    echo "Please install curl or wget to download the checkpoints." >&2
    exit 1
fi

for ckpt in "${CHECKPOINTS[@]}"; do
    if [ -f "$ckpt" ]; then
        echo "$ckpt already exists, skipping"
        continue
    fi
    echo "Downloading $ckpt..."
    download "$ckpt" "$BASE_URL/$ckpt"
done
