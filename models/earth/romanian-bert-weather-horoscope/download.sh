#!/usr/bin/env bash

set -euo pipefail

# 项目根目录，无论用户从哪个位置运行脚本都能正确定位
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WEIGHT_DIR="${PROJECT_DIR}/weight"
WEIGHT_FILE="${WEIGHT_DIR}/model.safetensors"

HF_URL="https://huggingface.co/iulik-pisik/romanian-bert-weather-horoscope/resolve/main/model.safetensors?download=true"

mkdir -p "${WEIGHT_DIR}"

echo "============================================================"
echo "开始下载 Romanian BERT 模型权重"
echo "保存位置：${WEIGHT_FILE}"
echo "============================================================"

# -L：跟随重定向
# -f：下载失败时返回错误
# -C -：支持断点续传
# --retry：下载异常时自动重试
curl -fL \
  --retry 5 \
  --retry-delay 2 \
  -C - \
  -o "${WEIGHT_FILE}" \
  "${HF_URL}"

echo
echo "权重下载完成："
ls -lh "${WEIGHT_FILE}"

echo
echo "检查 Safetensors 文件是否能正常读取……"

python - "${WEIGHT_FILE}" <<'PY'
import os
import sys
from safetensors import safe_open

file_path = sys.argv[1]
size_mb = os.path.getsize(file_path) / 1024**2

with safe_open(file_path, framework="pt", device="cpu") as f:
    tensor_names = list(f.keys())

print(f"权重文件大小：{size_mb:.2f} MB")
print(f"张量数量：{len(tensor_names)}")
print("Safetensors 权重文件检查成功")
PY

echo
echo "============================================================"
echo "模型权重已成功下载到 weight/model.safetensors"
echo "现在可以运行：python scripts/test.py"
echo "============================================================"
