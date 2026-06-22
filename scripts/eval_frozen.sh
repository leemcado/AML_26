#!/bin/bash
# scripts/eval_frozen.sh — Frozen test (hyperbolic structure observation)
#
# Usage:
#   bash scripts/eval_frozen.sh --ckpt results/.../checkpoints/0100000.pt --model DiT-S/2 --gpus 0
#   bash scripts/eval_frozen.sh --ckpt ... --model HiT-B/2 --gpus 1 --proj-ckpt same
#   bash scripts/eval_frozen.sh --ckpt ... --model DiT-B/2 --gpus 0 --latent-dir data/cached_latents

set -eo pipefail

CKPT=""
MODEL="DiT-B/2"
GPUS="0"
CURVATURE=1.0
NUM_SAMPLES=1000
LATENT_DIR=""
OUTPUT_DIR="frozen_test_results"
PROJ_CKPT=""
PROJ_DIM=256

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt) CKPT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --curvature) CURVATURE="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --latent-dir) LATENT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --proj-ckpt) PROJ_CKPT="$2"; shift 2 ;;
        --proj-dim) PROJ_DIM="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CKPT" ]]; then
    echo "Error: --ckpt required"
    exit 1
fi

# --proj-ckpt same: reuse the model checkpoint itself
if [[ "$PROJ_CKPT" == "same" ]]; then
    PROJ_CKPT="$CKPT"
fi

# Use first GPU only (single process)
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}"

ARGS="--model $MODEL --ckpt $CKPT --curvature $CURVATURE --num-samples $NUM_SAMPLES --output-dir $OUTPUT_DIR --proj-dim $PROJ_DIM"

if [[ -n "$LATENT_DIR" ]]; then
    ARGS="$ARGS --latent-dir $LATENT_DIR"
fi
if [[ -n "$PROJ_CKPT" ]]; then
    ARGS="$ARGS --proj-ckpt $PROJ_CKPT"
fi

echo "=========================================="
echo "  Frozen Test"
echo "  Model  : $MODEL"
echo "  Ckpt   : $CKPT"
echo "  GPU    : ${GPU_ARRAY[0]}"
echo "  Samples: $NUM_SAMPLES"
echo "=========================================="

python evaluation/frozen_test.py $ARGS
