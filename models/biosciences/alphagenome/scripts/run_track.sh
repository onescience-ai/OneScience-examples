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

mkdir -p "${PROJECT_DIR}/outputs_track"
python "${SCRIPT_DIR}/run_track_prediction_eval.py" \
    --model_dir "${MODEL_ROOT_DIR}/alphagenome-all-folds" \
    --model_version ALL_FOLDS \
    --data_dir "${DATA_ROOT_DIR}/v1/train" \
    --output_path "${PROJECT_DIR}/outputs_track/eval_results.csv"

