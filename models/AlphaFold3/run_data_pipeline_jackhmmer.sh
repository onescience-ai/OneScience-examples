#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.95}"
export TRITON_ENABLE_GLOBAL_TO_LOCAL="${TRITON_ENABLE_GLOBAL_TO_LOCAL:-1}"
export TRITON_USE_MAKE_BLOCK_PTR="${TRITON_USE_MAKE_BLOCK_PTR:-1}"
export TRITON_DEFAULT_ENABLE_NUM_VGPRS512="${TRITON_DEFAULT_ENABLE_NUM_VGPRS512:-1}"

MODEL_DIR="${MODEL_DIR:-checkpoints/AlphaFold3}"
DATASET_ROOT="${DATASET_ROOT:-data/alphafold3_dataset}"
INPUT_JSON="${1:-inputs/t1119_search.json}"
OUTPUT_DIR="${2:-outputs/jackhmmer_pipeline}"

mkdir -p "${OUTPUT_DIR}"

python run_alphafold.py \
  --json_path="${INPUT_JSON}" \
  --model_dir="${MODEL_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --run_data_pipeline=true \
  --run_inference=false \
  --flash_attention_implementation="${FLASH_ATTENTION_IMPLEMENTATION:-cutlass}" \
  --use_mmseqs=false \
  --use_mmseqs_gpu=false \
  --db_dir "${DATASET_ROOT}/public_databases" \
  --mmseqs_db_dir="${DATASET_ROOT}/mmseqsDB" \
  --small_bfd_database_path="${DATASET_ROOT}/jackhmmer_split/bfd-first_non_consensus_sequences.fasta@64" \
  --small_bfd_z_value=65928866 \
  --mgnify_database_path="${DATASET_ROOT}/jackhmmer_split/mgy_clusters_2022_05.fa@512" \
  --mgnify_z_value=623796864 \
  --uniprot_cluster_annot_database_path="${DATASET_ROOT}/jackhmmer_split/uniprot_cluster_annot_2021_04.fa@256" \
  --uniprot_cluster_annot_z_value=225619586 \
  --uniref90_database_path="${DATASET_ROOT}/jackhmmer_split/uniref90_2022_05.fa@128" \
  --uniref90_z_value=153742194 \
  --jackhmmer_n_cpu="${JACKHMMER_N_CPU:-2}" \
  --jackhmmer_max_parallel_shards="${JACKHMMER_MAX_PARALLEL_SHARDS:-1}" \
  --jackhmmer_max_threads="${JACKHMMER_MAX_THREADS:-8}" \
  --nhmmer_n_cpu="${NHMMER_N_CPU:-2}" \
  --nhmmer_max_parallel_shards="${NHMMER_MAX_PARALLEL_SHARDS:-1}" \
  --nhmmer_max_threads="${NHMMER_MAX_THREADS:-8}"
