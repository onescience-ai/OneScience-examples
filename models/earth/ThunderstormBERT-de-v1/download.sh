#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Stickmu/ThunderstormBERT-de-v1"
BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/main"

download_file() {
    local filename="$1"
    local expected_sha256="$2"

    if [ -f "$filename" ] && echo "${expected_sha256}  ${filename}" | sha256sum -c --status; then
        echo "${filename} 已存在且校验通过，跳过下载。"
        return
    fi

    echo "正在下载 ${filename} ..."
    curl \
        -fL \
        --retry 5 \
        --retry-delay 2 \
        --continue-at - \
        --output "${filename}.part" \
        "${BASE_URL}/${filename}?download=true"

    echo "${expected_sha256}  ${filename}.part" | sha256sum -c -
    mv "${filename}.part" "${filename}"
    echo "${filename} 下载完成。"
}

download_file \
    "model.safetensors" \
    "cbd6dc7130bf22927f5446db2827e19a31a95762890b34f15da8d059c1e1e568"

download_file \
    "spm.model" \
    "13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3"

download_file \
    "tokenizer.json" \
    "9ddbe35b768b22d6cd0d61a0a88ea3a188b7c00bbb7b825f9f247cf0b79c2365"

echo "全部模型文件下载并校验完成。"
