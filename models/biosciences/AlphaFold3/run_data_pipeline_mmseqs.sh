#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export TRITON_ENABLE_GLOBAL_TO_LOCAL="${TRITON_ENABLE_GLOBAL_TO_LOCAL:-1}"
export TRITON_USE_MAKE_BLOCK_PTR="${TRITON_USE_MAKE_BLOCK_PTR:-1}"
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512="${TRITON_DEFAULT_ENABLE_NUM_VGPRS512:-1}"

MODEL_DIR="${MODEL_DIR:-checkpoints/AlphaFold3}"
DATASET_ROOT="${DATASET_ROOT:-data/alphafold3_dataset}"
INPUT_JSON="${1:-inputs/t1119_search.json}"
OUTPUT_DIR="${2:-outputs/mmseqs_pipeline}"

export PATH="${ROOT_DIR}/${MODEL_DIR}/mmseqs/bin:${PATH}"
export LD_LIBRARY_PATH="${ROOT_DIR}/${MODEL_DIR}/mmseqs/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUTPUT_DIR}"

python run_alphafold.py \
  --json_path="${INPUT_JSON}" \
  --model_dir="${MODEL_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --run_data_pipeline=true \
  --run_inference=false \
  --flash_attention_implementation=triton \
  --db_dir "${DATASET_ROOT}/public_databases" \
  --mmseqs_db_dir "${DATASET_ROOT}/mmseqsDB" \
  --use_mmseqs=true \
  --use_mmseqs_gpu="${USE_MMSEQS_GPU:-true}" \
  --mmseqs_options="${MMSEQS_OPTIONS:---num-iterations 1 --db-load-mode 2 -a --max-seqs 10000 --prefilter-mode 3}"
