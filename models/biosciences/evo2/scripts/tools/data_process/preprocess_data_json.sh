#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/data/json/pretraining_or_both_phases/promoters}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/promoters/pretraining_data_promoters}"
TOKENIZER="${TOKENIZER:-CharLevelTokenizer}"

mkdir -p "${OUTPUT_DIR}"

echo "EVO2_PROJECT_ROOT: ${PROJECT_ROOT}"
echo "EVO2_JSON_INPUT_DIR: ${INPUT_DIR}"
echo "EVO2_JSON_OUTPUT_DIR: ${OUTPUT_DIR}"

for SPLIT in train test valid; do
    INPUT_FILE="${INPUT_DIR}/data_promoters_${SPLIT}_chunk1.jsonl.gz"
    OUTPUT_PREFIX="${OUTPUT_DIR}/data_promoters_${SPLIT}"

    echo "Processing ${INPUT_FILE} -> ${OUTPUT_PREFIX}"
    python "${PROJECT_ROOT}/scripts/tools/data_process/preprocess_data_json.py" \
        --input "${INPUT_FILE}" \
        --output-prefix "${OUTPUT_PREFIX}" \
        --tokenizer-type "${TOKENIZER}" \
        --dataset-impl mmap \
        --append-eod \
        --workers 8 \
        --log-interval 100
    echo "Finished processing split: ${SPLIT}"
done