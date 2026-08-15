#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELSCOPE_MODEL="${MODELSCOPE_MODEL:-OneScience/scGPT}"
MODELSCOPE_DATASET="${MODELSCOPE_DATASET:-OneScience/scGPT_datasets}"

echo "== scGPT assets download =="
echo "Package root: ${ROOT_DIR}"
echo "ModelScope model: ${MODELSCOPE_MODEL}"
echo "ModelScope dataset: ${MODELSCOPE_DATASET}"

if ! command -v modelscope >/dev/null 2>&1; then
    echo "ModelScope CLI was not found; installing modelscope..."
    "${PYTHON_BIN}" -m pip install modelscope
    hash -r
fi

if ! command -v modelscope >/dev/null 2>&1; then
    echo "ERROR: modelscope command was not found after installation." >&2
    echo "Please make sure the Python scripts directory is in PATH." >&2
    exit 1
fi

# ---- 权重：从模型仓库下载 weight/** ----
mkdir -p "${ROOT_DIR}/weight"

modelscope download \
    --model "${MODELSCOPE_MODEL}" \
    --include "weight/**" \
    --local_dir "${ROOT_DIR}"

missing_assets=()
for model_dir in \
    scGPT_human \
    scGPT_brain \
    scGPT_bc \
    scGPT_heart \
    scGPT_kidney \
    scGPT_lung \
    scGPT_pan_cancer \
    scGPT_CP \
    finetuned_scGPT_adamson; do
    for asset in args.json best_model.pt vocab.json; do
        if [[ ! -f "${ROOT_DIR}/weight/${model_dir}/${asset}" ]]; then
            missing_assets+=("weight/${model_dir}/${asset}")
        fi
    done
done

if (( ${#missing_assets[@]} > 0 )); then
    echo "ERROR: The following weight files are missing after download:" >&2
    printf '  - %s\n' "${missing_assets[@]}" >&2
    exit 1
fi

echo "scGPT weights are ready under ${ROOT_DIR}/weight."
echo "The default model directory is ${ROOT_DIR}/weight/scGPT_human;"
echo "override it with the SCGPT_MODEL_DIR environment variable if needed."

# ---- 数据集：从数据集仓库下载到 data/（SCGPT_SKIP_DATASET=1 可跳过）----
if [[ "${SCGPT_SKIP_DATASET:-0}" == "1" ]]; then
    echo "Skipping dataset download (SCGPT_SKIP_DATASET=1)."
    exit 0
fi

mkdir -p "${ROOT_DIR}/data"

modelscope download \
    --dataset "${MODELSCOPE_DATASET}" \
    --local_dir "${ROOT_DIR}/data"

for asset in \
    annotation_pancreas/demo_test.h5ad \
    annotation_pancreas/demo_train.h5ad; do
    if [[ ! -f "${ROOT_DIR}/data/${asset}" ]]; then
        echo "ERROR: dataset file data/${asset} is missing after download." >&2
        exit 1
    fi
done

echo "scGPT dataset is ready under ${ROOT_DIR}/data."
