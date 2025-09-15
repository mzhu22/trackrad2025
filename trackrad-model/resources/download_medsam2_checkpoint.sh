#!/bin/bash
# Downloads the MedSAM2 checkpoint (https://github.com/bowang-lab/MedSAM2) into this directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CKPT=MedSAM2_latest.pt
URL="https://huggingface.co/wanglab/MedSAM2/resolve/main/$CKPT"

if [ -f "$CKPT" ]; then
    echo "$CKPT already exists, skipping"
    exit 0
fi

echo "Downloading $CKPT..."
if command -v curl &> /dev/null; then
    curl -fL -o "$CKPT" "$URL"
elif command -v wget &> /dev/null; then
    wget -O "$CKPT" "$URL"
else
    echo "Please install curl or wget to download the checkpoint." >&2
    exit 1
fi
