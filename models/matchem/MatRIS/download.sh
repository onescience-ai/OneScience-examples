#!/bin/bash
# MatRIS 大文件下载脚本
# 本仓库为轻量化示例仓库，weight/ 和 cif_file/ 需从 ModelScope 完整模型包下载。
set -e

MODEL_ID="OneScience/MatRIS"
TMP_DIR=$(mktemp -d)

echo "==> 正在从 ModelScope 下载 MatRIS 大文件: ${MODEL_ID}"
if ! command -v modelscope &> /dev/null; then
    echo "错误：未找到 modelscope 命令，请先安装 modelscope: pip install modelscope"
    exit 1
fi

modelscope download --model "${MODEL_ID}" --local_dir "${TMP_DIR}"

echo "==> 复制 weight/ 和 cif_file/ 到当前目录"
mkdir -p weight cif_file

if [ -d "${TMP_DIR}/weight" ]; then
    cp -r "${TMP_DIR}/weight"/* ./weight/
fi

if [ -d "${TMP_DIR}/cif_file" ]; then
    cp -r "${TMP_DIR}/cif_file"/* ./cif_file/
fi

rm -rf "${TMP_DIR}"

echo "==> 完成。weight/ 和 cif_file/ 已就绪。"
