#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_DIR="${CKPT_DIR:-${EVO2_CKPT_DIR:-${PROJECT_ROOT}/checkpoints/evo2_nemo_7b}}"
PROMPT="${PROMPT:-ATGCGT}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/evo2_generation.txt}"
RUN_WITH_SRUN="${RUN_WITH_SRUN:-0}"

if [[ -z "${CKPT_DIR}" ]]; then
    echo "ERROR: Evo2 checkpoint path is required. Put it under checkpoints/evo2_nemo_7b or set EVO2_CKPT_DIR/CKPT_DIR." >&2
    exit 1
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "ERROR: Evo2 checkpoint directory does not exist: ${CKPT_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "EVO2_PROJECT_ROOT: ${PROJECT_ROOT}"
echo "EVO2_CKPT_DIR: ${CKPT_DIR}"
echo "EVO2_OUTPUT_FILE: ${OUTPUT_FILE}"

if [ "${RUN_WITH_SRUN}" = "1" ]; then
    srun python scripts/infer.py \
        --ckpt-dir "${CKPT_DIR}" \
        --prompt "${PROMPT}" \
        --output-file "${OUTPUT_FILE}"
else
    python scripts/infer.py \
        --ckpt-dir "${CKPT_DIR}" \
        --prompt "${PROMPT}" \
        --output-file "${OUTPUT_FILE}"
fi

echo "Inference completed. Result file: ${OUTPUT_FILE}"
