#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ESM_WEIGHT_DIR="${ESM_WEIGHT_DIR:-${PROJECT_ROOT}/weight}"
ESM_OUTPUT_DIR="${ESM_OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
ESM_FASTA="${ESM_FASTA:-${PROJECT_ROOT}/data/fasta/few_proteins.fasta}"
ESM2_8M_WEIGHT="${ESM2_8M_WEIGHT:-${ESM_WEIGHT_DIR}/checkpoints/esm2_t6_8M_UR50D.pt}"

mkdir -p "${ESM_OUTPUT_DIR}"

if [[ ! -f "${ESM2_8M_WEIGHT}" ]]; then
  echo "Missing ${ESM2_8M_WEIGHT}"
  echo "Run: bash scripts/download_weights.sh ${ESM_WEIGHT_DIR}"
  exit 1
fi

python scripts/extract.py \
  "${ESM2_8M_WEIGHT}" \
  "${ESM_FASTA}" \
  "${ESM_OUTPUT_DIR}/embeddings" \
  --include mean per_tok \
  --repr_layers 6

if [[ "${RUN_ESMFOLD:-0}" == "1" ]]; then
  python scripts/fold.py \
    -i "${ESM_FASTA}" \
    -o "${ESM_OUTPUT_DIR}/pdb" \
    --model-dir "${ESM_WEIGHT_DIR}" \
    --cpu-only
fi

if [[ "${RUN_VARIANT_PREDICTION:-0}" == "1" ]]; then
  python scripts/variant_prediction/predict.py \
    --model-location esm1v_t33_650M_UR90S_1 \
    --sequence "${ESM_VARIANT_SEQUENCE:?Set ESM_VARIANT_SEQUENCE}" \
    --dms-input data/variant_prediction/BLAT_ECOLX_Ranganathan2015.csv \
    --mutation-col mutant \
    --dms-output "${ESM_OUTPUT_DIR}/variant_prediction.csv" \
    --offset-idx 24 \
    --scoring-strategy wt-marginals
fi
