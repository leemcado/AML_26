#!/bin/bash
# scripts/eval_fid.sh — FID measurement
#
# Usage:
#   bash scripts/eval_fid.sh --ckpt results/.../checkpoints/0100000.pt --model DiT-S/2 --gpus 0,1
#   bash scripts/eval_fid.sh --ckpt ... --model DiT-S/4 --gpus 0 --ref-path /path/to/imagenet_val
#   bash scripts/eval_fid.sh --model DiT-S/2 --gpus 0 --dummy

set -eo pipefail

CKPT=""
MODEL="DiT-B/2"
GPUS="0"
REF_PATH=""
NUM_SAMPLES=50000
STEPS=250
CFG=4.0
OUTPUT_DIR=""
EXTRA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt) CKPT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --ref-path) REF_PATH="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --cfg) CFG="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dummy) EXTRA="$EXTRA --dummy"; shift ;;
        --no-compile) EXTRA="$EXTRA --no-compile"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    MODEL_SLUG=$(echo "$MODEL" | tr '/' '-')
    OUTPUT_DIR="fid_samples_${MODEL_SLUG}"
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}

export CUDA_VISIBLE_DEVICES="$GPUS"

ARGS="--model $MODEL --num-fid-samples $NUM_SAMPLES --num-sampling-steps $STEPS --cfg-scale $CFG --output-dir $OUTPUT_DIR"

if [[ -n "$CKPT" ]]; then
    ARGS="$ARGS --ckpt $CKPT"
fi
if [[ -n "$REF_PATH" ]]; then
    ARGS="$ARGS --ref-path $REF_PATH"
fi

echo "=========================================="
echo "  FID Evaluation"
echo "  Model  : $MODEL"
echo "  Ckpt   : ${CKPT:-none}"
echo "  GPUs   : $GPUS ($NPROC)"
echo "  Samples: $NUM_SAMPLES"
echo "=========================================="

MASTER_PORT=$((29500 + RANDOM % 1000))

if [[ $NPROC -gt 1 ]]; then
    torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT evaluation/fid.py $ARGS $EXTRA
else
    python evaluation/fid.py $ARGS $EXTRA
fi
