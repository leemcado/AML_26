#!/bin/bash
# scripts/eval_interp.sh — Interpolation experiment (Euclidean / Geodesic / SLERP / +Denoise)
#
# Usage:
#   bash scripts/eval_interp.sh --ckpt results/.../checkpoints/0100000.pt --model DiT-S/2 --gpus 0
#   bash scripts/eval_interp.sh --ckpt ... --model HiT-B/2 --gpus 1 --same-class 207 --cross 207,949

set -eo pipefail

CKPT=""
MODEL="DiT-B/2"
GPUS="0"
CURVATURE=1.0
CFG=4.0
STEPS=50
SEED=42
SAME_CLASS=207
CROSS=""
OUTPUT_DIR="interpolation_results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt) CKPT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --curvature) CURVATURE="$2"; shift 2 ;;
        --cfg) CFG="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --same-class) SAME_CLASS="$2"; shift 2 ;;
        --cross) CROSS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CKPT" ]]; then
    echo "Error: --ckpt required"
    exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}"

ARGS="--ckpt $CKPT --model $MODEL --curvature $CURVATURE --cfg-scale $CFG --num-sampling-steps $STEPS --seed $SEED --same-class $SAME_CLASS --output-dir $OUTPUT_DIR"

if [[ -n "$CROSS" ]]; then
    ARGS="$ARGS --cross-classes $CROSS"
fi

echo "=========================================="
echo "  Interpolation Compare"
echo "  Model  : $MODEL"
echo "  Ckpt   : $CKPT"
echo "  GPU    : ${GPU_ARRAY[0]}"
echo "  Class  : same=$SAME_CLASS, cross=${CROSS:-none}"
echo "=========================================="

python evaluation/interpolation_compare.py $ARGS
