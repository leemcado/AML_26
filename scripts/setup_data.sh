#!/bin/bash
# scripts/setup_data.sh — Prepare training data
#
# Usage:
#   bash scripts/setup_data.sh --source ~/HiT/data/cached_latents     # symlink (same server)
#   bash scripts/setup_data.sh --download                              # HuggingFace download

set -eo pipefail

SOURCE=""
DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --source) SOURCE="$2"; shift 2 ;;
        --download) DOWNLOAD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TARGET="data/cached_latents"

if [[ -d "$TARGET" || -L "$TARGET" ]]; then
    echo "Already exists: $TARGET"
    ls -ld "$TARGET"
    exit 0
fi

if [[ -n "$SOURCE" ]]; then
    if [[ ! -d "$SOURCE" ]]; then
        echo "Error: source directory not found: $SOURCE"
        exit 1
    fi
    mkdir -p data
    ln -s "$(realpath "$SOURCE")" "$TARGET"
    echo "Symlink created: $TARGET -> $(realpath "$SOURCE")"

elif [[ "$DOWNLOAD" == true ]]; then
    echo "Downloading from HuggingFace..."
    python data/download_data.py --dataset imagenet-1k --output-dir "$TARGET"

else
    echo "Error: --source <path> or --download required"
    echo ""
    echo "Usage:"
    echo "  bash scripts/setup_data.sh --source ~/HiT/data/cached_latents"
    echo "  bash scripts/setup_data.sh --download"
    exit 1
fi

echo "Done. Number of classes: $(ls "$TARGET" | wc -l)"
