#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${ONESCIENCE_DATASETS_DIR:-}" ]]; then
  DATA_ROOT_DIR="${ONESCIENCE_DATASETS_DIR}/AlphaGenome"
else
  DATA_ROOT_DIR="${PROJECT_DIR}/data"
fi

if [[ -n "${ONESCIENCE_MODELS_DIR:-}" ]]; then
  MODEL_ROOT_DIR="${ONESCIENCE_MODELS_DIR}/AlphaGenome"
else
  MODEL_ROOT_DIR="${PROJECT_DIR}/weight"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/run_variant_scoring.py" \
    --fasta_path "${DATA_ROOT_DIR}/reference/HOMO_SAPIENS/GRCh38.p13.genome.fa" \
    --model_dir "${MODEL_ROOT_DIR}/alphagenome-all-folds" \
    --output_dir "${PROJECT_DIR}/outputs_variant"
