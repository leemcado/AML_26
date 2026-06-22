#!/bin/bash
# scripts/train.sh — DiT/HiT training launcher
#
# Usage:
#   bash scripts/train.sh --config configs/dit_s_p2.yaml --gpus 0,1
#   bash scripts/train.sh --config configs/hit_b.yaml --gpus 0,1,2,3
#   bash scripts/train.sh --config configs/dit_s_p4.yaml --gpus 2  # single GPU
#   bash scripts/train.sh --config configs/dit_s_p2.yaml --gpus 0,1 --dummy
#
# Options:
#   --config   YAML config file path (required)
#   --gpus     GPU IDs to use (comma-separated, default: 0)
#   --dummy    Test with dummy data
#   --resume   Resume training from checkpoint

set -eo pipefail

CONFIG=""
GPUS="0"
DUMMY=""
RESUME=""
MAX_STEPS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --dummy) DUMMY="--dummy"; shift ;;
        --resume) RESUME="--resume $2"; shift 2 ;;
        --max-steps) MAX_STEPS="--max-steps $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config required"
    echo "Usage: bash scripts/train.sh --config configs/dit_s_p2.yaml --gpus 0,1"
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}

# Auto-detect HiT vs DiT from config filename
if [[ "$CONFIG" == *"hit"* ]]; then
    SCRIPT="training/train_hit.py"
else
    SCRIPT="training/train_dit.py"
fi

export CUDA_VISIBLE_DEVICES="$GPUS"

echo "=========================================="
echo "  Config : $CONFIG"
echo "  Script : $SCRIPT"
echo "  GPUs   : $GPUS ($NPROC)"
echo "=========================================="

MASTER_PORT=$((29500 + RANDOM % 1000))

if [[ $NPROC -gt 1 ]]; then
    torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT $SCRIPT \
        --config "$CONFIG" $DUMMY $RESUME $MAX_STEPS
else
    python $SCRIPT \
        --config "$CONFIG" $DUMMY $RESUME $MAX_STEPS
fi
