#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/data_mini}"
GENOME_DIR="${GENOME_DIR:-${DATA_ROOT}/genome_data}"

mkdir -p "${GENOME_DIR}"
cd "${GENOME_DIR}"

echo "EVO2_PROJECT_ROOT: ${PROJECT_ROOT}"
echo "EVO2_GENOME_DIR: ${GENOME_DIR}"

for chr in chr20 chr21 chr22; do
    if [ ! -f "${chr}.fa.gz" ]; then
        wget -c "https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/${chr}.fa.gz"
    fi
    if [ ! -f "${chr}.fa" ]; then
        zcat "${chr}.fa.gz" > "${chr}.fa"
    fi
done

cat chr20.fa chr21.fa chr22.fa > chr20_21_22.fa

python "${PROJECT_ROOT}/scripts/tools/data_process/preprocess_data_fasta.py" \
    --config "${PROJECT_ROOT}/config/genome_preprocess_config.yaml"

echo "FASTA preprocessing completed: ${GENOME_DIR}/preprocessed_data"