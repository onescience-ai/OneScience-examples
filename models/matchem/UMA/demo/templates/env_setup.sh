#!/bin/bash
# env_setup.sh - shared environment setup for UMA demo
# Usage: source templates/env_setup.sh <conda_env> <module1> [module2 ...]

CONDA_ENV="${1:?Please provide conda environment name}"
shift

# System scripts (bashrc/conda/module) may reference unset vars.
# Temporarily relax bash strict mode during bootstrap.
set +eu

module purge
source ~/.bashrc
conda activate "$CONDA_ENV"

for mod in "$@"; do
    module load "$mod"
done

set -eu

# Load shared dataset/model roots from repo env.sh if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
if [ -f "$REPO_ROOT/env.sh" ]; then
    source "$REPO_ROOT/env.sh"
else
    # Fallback: point dataset/model roots to repo root
    export ONESCIENCE_DATASETS_DIR="${ONESCIENCE_DATASETS_DIR:-$REPO_ROOT}"
    export ONESCIENCE_MODELS_DIR="${ONESCIENCE_MODELS_DIR:-$REPO_ROOT}"
fi

# UMA-specific runtime asset: Jd.pt
# 如仓库根目录 weight/ 下存在 Jd.pt，会自动设置；否则仅打印警告，不影响其他推理。
export ONESCIENCE_UMA_JD_PATH="${ONESCIENCE_MODELS_DIR}/weight/Jd.pt"

if [ -f "${ONESCIENCE_UMA_JD_PATH}" ]; then
    echo "[OK] 文件: ${ONESCIENCE_UMA_JD_PATH}"
else
    echo "[WARN] 文件不存在: ${ONESCIENCE_UMA_JD_PATH}"
fi

