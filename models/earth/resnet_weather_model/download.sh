#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="sallyanndelucia/resnet_weather_model"
FILENAME="pytorch_model.bin"
EXPECTED_SHA256="24637dd6656f51d1beccb47e6710d7aa963b48b4b06f861516de39f6fd537efe"
DOWNLOAD_URL="https://huggingface.co/${MODEL_ID}/resolve/main/${FILENAME}?download=true"

if [ -f "$FILENAME" ] && \
   echo "${EXPECTED_SHA256}  ${FILENAME}" | sha256sum -c --status; then
    echo "${FILENAME} 已存在且校验通过，跳过下载。"
    exit 0
fi

echo "正在从Hugging Face下载 ${FILENAME} ..."

curl \
    -fL \
    --retry 5 \
    --retry-delay 2 \
    --continue-at - \
    --output "${FILENAME}.part" \
    "$DOWNLOAD_URL"

echo "${EXPECTED_SHA256}  ${FILENAME}.part" | sha256sum -c -

mv "${FILENAME}.part" "${FILENAME}"

echo "${FILENAME} 下载完成并通过SHA256校验。"
