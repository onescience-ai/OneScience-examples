#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${PROJECT_ROOT}/../onescience/env.sh" ]; then
  # Optional: reuse OneScience environment variables when this split repo lives
  # next to the full OneScience checkout.
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/../onescience/env.sh"
fi

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TRITON_ENABLE_GLOBAL_TO_LOCAL="${TRITON_ENABLE_GLOBAL_TO_LOCAL:-1}"
export TRITON_USE_MAKE_BLOCK_PTR="${TRITON_USE_MAKE_BLOCK_PTR:-1}"
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512="${TRITON_DEFAULT_ENABLE_NUM_VGPRS512:-1}"

MODEL_ROOT="${ONESCIENCE_MODELS_DIR:-${PROJECT_ROOT}/weight}"
MODEL_DIR="${ALPHAFOLD3_MODEL_DIR:-${MODEL_ROOT}}"
JSON_PATH="${ALPHAFOLD3_JSON_PATH:-${PROJECT_ROOT}/inputs/7r6r_data.json}"
OUTPUT_DIR="${ALPHAFOLD3_OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
FLASH_ATTENTION="${ALPHAFOLD3_FLASH_ATTENTION:-triton}"

mkdir -p "${OUTPUT_DIR}"

python "${PROJECT_ROOT}/scripts/run_alphafold.py" \
  --json_path="${JSON_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --run_data_pipeline=false \
  --flash_attention_implementation="${FLASH_ATTENTION}"
