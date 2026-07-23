#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://huggingface.co/extreme-weather-impacts/environmentalBERT-extremeweather/resolve/main"

FILES=(
    "model.safetensors"
    "tokenizer.json"
    "vocab.json"
    "merges.txt"
)

for FILE_NAME in "${FILES[@]}"; do
    OUTPUT_FILE="${PROJECT_DIR}/${FILE_NAME}"
    URL="${BASE_URL}/${FILE_NAME}?download=true"

    echo "============================================================"
    echo "正在下载：${FILE_NAME}"

    curl -fL \
      --retry 5 \
      --retry-delay 2 \
      -C - \
      -o "${OUTPUT_FILE}" \
      "${URL}"

    if [ ! -s "${OUTPUT_FILE}" ]; then
        echo "下载失败或文件为空：${FILE_NAME}"
        exit 1
    fi

    ls -lh "${OUTPUT_FILE}"
done

echo "============================================================"
echo "模型权重和Tokenizer文件全部下载完成"
