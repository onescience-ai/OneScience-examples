#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TRITON_ENABLE_GLOBAL_TO_LOCAL="${TRITON_ENABLE_GLOBAL_TO_LOCAL:-1}"
export TRITON_USE_MAKE_BLOCK_PTR="${TRITON_USE_MAKE_BLOCK_PTR:-1}"
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512="${TRITON_DEFAULT_ENABLE_NUM_VGPRS512:-1}"

INPUT_JSON="${1:-inputs/7r6r_data.json}"
OUTPUT_DIR="${2:-outputs/msa_inference}"
MODEL_DIR="${MODEL_DIR:-checkpoints/AlphaFold3}"
FLASH_IMPL="${FLASH_ATTENTION_IMPLEMENTATION:-triton}"

mkdir -p "${OUTPUT_DIR}"

python run_alphafold.py \
  --json_path="${INPUT_JSON}" \
  --model_dir="${MODEL_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --run_data_pipeline=false \
  --flash_attention_implementation="${FLASH_IMPL}"
