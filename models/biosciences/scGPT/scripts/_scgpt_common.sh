#!/usr/bin/env bash

set -euo pipefail

SCGPT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCGPT_PROJECT_ROOT="$(cd "${SCGPT_SCRIPT_DIR}/.." && pwd)"

# DCU 环境：加载 DTK 运行时环境（NVIDIA GPU 环境可忽略）。
if [[ -n "${ROCM_PATH:-}" && -r "${ROCM_PATH}/cuda/env.sh" ]]; then
    source "${ROCM_PATH}/cuda/env.sh"
fi

# 优先使用当前 conda 环境中的 Python：DCU 的 env.sh 会把 DTK bin 前置到
# PATH，可能遮蔽 `python`，故在 CONDA_PREFIX 可用时直接使用其绝对路径。
SCGPT_PYTHON="${SCGPT_PYTHON:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}"
SCGPT_PYTHON="${SCGPT_PYTHON:-python}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    SCGPT_SITE_PACKAGES="$("${SCGPT_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${SCGPT_SITE_PACKAGES}/fastpt/torch/lib:${LD_LIBRARY_PATH:-}"
fi

# 模型与数据路径（均可用环境变量覆盖，见 README.md 与 config/config.yaml）。
SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-${SCGPT_PROJECT_ROOT}/weight/scGPT_human}"
SCGPT_DATASET_ROOT="${SCGPT_DATASET_ROOT:-${SCGPT_PROJECT_ROOT}/data}"
SCGPT_INFERENCE_DATA="${SCGPT_INFERENCE_DATA:-${SCGPT_DATASET_ROOT}/annotation_pancreas/demo_test.h5ad}"
SCGPT_FINETUNE_DATA="${SCGPT_FINETUNE_DATA:-${SCGPT_DATASET_ROOT}/annotation_pancreas/demo_train.h5ad}"
SCGPT_OUTPUT_ROOT="${SCGPT_OUTPUT_ROOT:-${SCGPT_PROJECT_ROOT}/outputs}"
SCGPT_DEVICE="${SCGPT_DEVICE:-cuda}"
SCGPT_TORCHRUN="${SCGPT_TORCHRUN:-torchrun}"

# 项目根目录提供本地 `model` 包（自包含，不依赖 onescience）；
# 数据管线等公共能力来自环境中已安装的 onescience 包。
export PYTHONPATH="${SCGPT_PROJECT_ROOT}:${PYTHONPATH:-}"

scgpt_require_file() {
    local path="$1"
    if [[ ! -r "${path}" ]]; then
        echo "Required file is not readable: ${path}" >&2
        exit 2
    fi
}

scgpt_require_model() {
    scgpt_require_file "${SCGPT_MODEL_DIR}/args.json"
    scgpt_require_file "${SCGPT_MODEL_DIR}/best_model.pt"
    scgpt_require_file "${SCGPT_MODEL_DIR}/vocab.json"
}

scgpt_launch() {
    local program="$1"
    local num_devices
    shift
    num_devices="$("${SCGPT_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
    if [[ "${SCGPT_DEVICE}" == cuda* && "${num_devices}" -lt 1 ]]; then
        echo "No CUDA/DTK device is visible to PyTorch" >&2
        exit 2
    fi
    if [[ "${SCGPT_DEVICE}" == cuda* && "${num_devices}" -gt 1 ]]; then
        echo "Detected ${num_devices} visible devices; launching distributed scGPT"
        exec "${SCGPT_TORCHRUN}" \
            --standalone \
            --nproc-per-node "${num_devices}" \
            "${program}" "$@"
    fi
    exec "${SCGPT_PYTHON}" "${program}" "$@"
}

mkdir -p "${SCGPT_OUTPUT_ROOT}"
cd "${SCGPT_PROJECT_ROOT}"
