#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${PROJECT_ROOT}/../onescience/env.sh" ]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/../onescience/env.sh"
fi
if [ -n "${ROCM_PATH:-}" ] && [ -f "${ROCM_PATH}/cuda/env.sh" ]; then
  # shellcheck source=/dev/null
  source "${ROCM_PATH}/cuda/env.sh"
fi

MODEL_ROOT="${ONESCIENCE_MODELS_DIR:-${PROJECT_ROOT}/weight}"
MODEL_DIR="${ALPHAFOLD3_MODEL_DIR:-${MODEL_ROOT}}"
MMSEQS_HOME="${ALPHAFOLD3_MMSEQS_HOME:-${MODEL_DIR}/mmseqs}"
export PATH="${MMSEQS_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MMSEQS_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TRITON_ENABLE_GLOBAL_TO_LOCAL="${TRITON_ENABLE_GLOBAL_TO_LOCAL:-1}"
export TRITON_USE_MAKE_BLOCK_PTR="${TRITON_USE_MAKE_BLOCK_PTR:-1}"
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512="${TRITON_DEFAULT_ENABLE_NUM_VGPRS512:-1}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

DATASET_ROOT="${ALPHAFOLD3_DATASET_ROOT:-${ONESCIENCE_DATASETS_DIR:-${PROJECT_ROOT}/data}/alphafold3}"
DB_DIRS="${ALPHAFOLD3_DB_DIR:-${DATASET_ROOT}/public_databases}"
MMSEQS_DB_DIRS="${ALPHAFOLD3_MMSEQS_DB_DIR:-${DATASET_ROOT}/mmseqsDB}"
JSON_PATH="${ALPHAFOLD3_JSON_PATH:-${PROJECT_ROOT}/inputs/t1119_search.json}"
OUTPUT_DIR="${ALPHAFOLD3_OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
RUN_INFERENCE="${ALPHAFOLD3_RUN_INFERENCE:-false}"
FLASH_ATTENTION="${ALPHAFOLD3_FLASH_ATTENTION:-triton}"
MMSEQS_OPTIONS="${ALPHAFOLD3_MMSEQS_OPTIONS:---num-iterations 1 --db-load-mode 2 -a --max-seqs 10000 --prefilter-mode 3}"

mkdir -p "${OUTPUT_DIR}"

python "${PROJECT_ROOT}/scripts/run_alphafold.py" \
  --json_path="${JSON_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --run_data_pipeline=true \
  --run_inference="${RUN_INFERENCE}" \
  --flash_attention_implementation="${FLASH_ATTENTION}" \
  --db_dir="${DB_DIRS}" \
  --mmseqs_db_dir="${MMSEQS_DB_DIRS}" \
  --use_mmseqs=true \
  --use_mmseqs_gpu=true \
  --mmseqs_options="${MMSEQS_OPTIONS}"
