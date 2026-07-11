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

export PATH="${PROJECT_ROOT}/flax_model/alphafold3/_tools/hmmer/bin:${PATH}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.95}"
export TRITON_ENABLE_GLOBAL_TO_LOCAL="${TRITON_ENABLE_GLOBAL_TO_LOCAL:-1}"
export TRITON_USE_MAKE_BLOCK_PTR="${TRITON_USE_MAKE_BLOCK_PTR:-1}"
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512="${TRITON_DEFAULT_ENABLE_NUM_VGPRS512:-1}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

MODEL_ROOT="${ONESCIENCE_MODELS_DIR:-${PROJECT_ROOT}/weight}"
MODEL_DIR="${ALPHAFOLD3_MODEL_DIR:-${MODEL_ROOT}}"
DATASET_ROOT="${ALPHAFOLD3_DATASET_ROOT:-${ONESCIENCE_DATASETS_DIR:-${PROJECT_ROOT}/data}/alphafold3}"
DB_DIRS="${ALPHAFOLD3_DB_DIR:-${DATASET_ROOT}/public_databases}"
MMSEQS_DB_DIRS="${ALPHAFOLD3_MMSEQS_DB_DIR:-${DATASET_ROOT}/mmseqsDB}"
JSON_PATH="${ALPHAFOLD3_JSON_PATH:-${PROJECT_ROOT}/inputs/t1119_search.json}"
OUTPUT_DIR="${ALPHAFOLD3_OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
RUN_INFERENCE="${ALPHAFOLD3_RUN_INFERENCE:-false}"
FLASH_ATTENTION="${ALPHAFOLD3_FLASH_ATTENTION:-cutlass}"

mkdir -p "${OUTPUT_DIR}"

python "${PROJECT_ROOT}/scripts/run_alphafold.py" \
  --json_path="${JSON_PATH}" \
  --model_dir="${MODEL_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --run_data_pipeline=true \
  --run_inference="${RUN_INFERENCE}" \
  --flash_attention_implementation="${FLASH_ATTENTION}" \
  --use_mmseqs=false \
  --use_mmseqs_gpu=false \
  --db_dir="${DB_DIRS}" \
  --mmseqs_db_dir="${MMSEQS_DB_DIRS}" \
  --small_bfd_database_path="${DATASET_ROOT}/jackhmmer_split/bfd-first_non_consensus_sequences.fasta@64" \
  --small_bfd_z_value=65928866 \
  --mgnify_database_path="${DATASET_ROOT}/jackhmmer_split/mgy_clusters_2022_05.fa@512" \
  --mgnify_z_value=623796864 \
  --uniprot_cluster_annot_database_path="${DATASET_ROOT}/jackhmmer_split/uniprot_cluster_annot_2021_04.fa@256" \
  --uniprot_cluster_annot_z_value=225619586 \
  --uniref90_database_path="${DATASET_ROOT}/jackhmmer_split/uniref90_2022_05.fa@128" \
  --uniref90_z_value=153742194 \
  --jackhmmer_n_cpu="${ALPHAFOLD3_JACKHMMER_N_CPU:-2}" \
  --jackhmmer_max_parallel_shards="${ALPHAFOLD3_JACKHMMER_MAX_PARALLEL_SHARDS:-1}" \
  --jackhmmer_max_threads="${ALPHAFOLD3_JACKHMMER_MAX_THREADS:-8}" \
  --nhmmer_n_cpu="${ALPHAFOLD3_NHMMER_N_CPU:-2}" \
  --nhmmer_max_parallel_shards="${ALPHAFOLD3_NHMMER_MAX_PARALLEL_SHARDS:-1}" \
  --nhmmer_max_threads="${ALPHAFOLD3_NHMMER_MAX_THREADS:-8}"
