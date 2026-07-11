#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_DIR="${DATASET_DIR:-${EVO2_DATASET_DIR:-${PROJECT_ROOT}/data/data_mini/genome_data}}"
CKPT_DIR="${CKPT_DIR:-}"
AVAILABLE_DEVICES="$(python -c "import torch; c=torch.cuda.device_count(); print(c if c > 0 else 1)")"
DEVICES="${DEVICES:-${AVAILABLE_DEVICES}}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"
MAX_STEPS="${MAX_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-50}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-2}"
WORKERS="${WORKERS:-8}"
CHECKPOINTING="${CHECKPOINTING:-1}"

EXTRA_ARGS=()
if [[ -n "${GLOBAL_BATCH_SIZE}" ]]; then
    EXTRA_ARGS+=(--global-batch-size "${GLOBAL_BATCH_SIZE}")
fi
if [[ "${CHECKPOINTING}" != "1" ]]; then
    EXTRA_ARGS+=(--disable-checkpointing --no-save-last-checkpoint)
fi

if [[ -n "${CKPT_DIR}" ]]; then
    if [[ ! -d "${CKPT_DIR}" ]]; then
        echo "Error: Evo2 1B checkpoint directory not found: ${CKPT_DIR}" >&2
        echo "Set CKPT_DIR only when you want to restore from a 1B NeMo checkpoint." >&2
        exit 1
    fi
    EXTRA_ARGS+=(--ckpt-dir "${CKPT_DIR}")
fi

echo "EVO2_PROJECT_ROOT: ${PROJECT_ROOT}"
echo "EVO2_DATASET_DIR: ${DATASET_DIR}"
echo "EVO2_CKPT_DIR: ${CKPT_DIR:-<none, train from scratch>}"
echo "EVO2_DEVICES: ${DEVICES}"
echo "EVO2_SEQ_LENGTH: ${SEQ_LENGTH}"
echo "EVO2_MAX_STEPS: ${MAX_STEPS}"
echo "EVO2_CHECKPOINTING: ${CHECKPOINTING}"
echo "EVO2_WORKERS: ${WORKERS}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

python scripts/train.py \
    -d config/genome_data_config.yaml \
    --dataset-dir "${DATASET_DIR}" \
    --model-size 1b \
    --devices "${DEVICES}" \
    --num-nodes 1 \
    --seq-length "${SEQ_LENGTH}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --lr 0.0001 \
    --warmup-steps "${WARMUP_STEPS}" \
    --max-steps "${MAX_STEPS}" \
    --clip-grad 1 \
    --wd 0.01 \
    --workers "${WORKERS}" \
    --activation-checkpoint-recompute-num-layers 1 \
    --val-check-interval "${VAL_CHECK_INTERVAL}" \
    --limit-val-batches "${LIMIT_VAL_BATCHES}" \
    "${EXTRA_ARGS[@]}"
