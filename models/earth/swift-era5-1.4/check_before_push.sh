#!/usr/bin/env bash
set -euo pipefail

# 在 git add / commit / push 前运行。
# 有暂存文件时检查整个暂存区；暂存区为空时只检查本脚本所在模型目录。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOCKED_REGEX='(^|/)(weights|sample-data|checkpoints|logs?|outputs?|results?|runs|wandb|tensorboard|venv|\.venv|__pycache__|\.ipynb_checkpoints)(/|$)|\.(log|out|err|pt|pth|ckpt|safetensors|onnx|h5|hdf5|npz|npy|part|zip|tar|tgz|7z|rar)$'
MAX_BYTES=$((50 * 1024 * 1024))
ERRORS=0
declare -a FILES=()

if git -C "${SCRIPT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
    while IFS= read -r -d '' file; do
        FILES+=("${file}")
    done < <(git -C "${REPO_ROOT}" diff --cached --name-only --diff-filter=ACMR -z)

    if (( ${#FILES[@]} > 0 )); then
        CHECK_ROOT="${REPO_ROOT}"
        echo "[提示] 正在检查整个暂存区中的 ${#FILES[@]} 个文件。"
    else
        CHECK_ROOT="${SCRIPT_DIR}"
        echo "[提示] 暂存区为空，改为检查当前模型目录。"
    fi
else
    CHECK_ROOT="${SCRIPT_DIR}"
    echo "[提示] 当前不在 Git 仓库中，检查当前模型目录。"
fi

if (( ${#FILES[@]} == 0 )); then
    while IFS= read -r -d '' file; do
        FILES+=("${file#./}")
    done < <(cd "${CHECK_ROOT}" && find . -type f -print0 ! -path './.git/*')
fi

for file in "${FILES[@]}"; do
    full_path="${CHECK_ROOT}/${file}"
    [[ -f "${full_path}" ]] || continue

    if [[ "${file}" =~ ${BLOCKED_REGEX} ]]; then
        echo "[禁止上传] ${file}"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    size="$(stat -c '%s' "${full_path}")"
    if (( size > MAX_BYTES )); then
        echo "[大文件] ${file} ($((size / 1024 / 1024)) MiB，超过 50 MiB)"
        ERRORS=$((ERRORS + 1))
    fi
done

echo "[提示] 检查模型目录中的常见明文令牌……"
SECRET_MATCHES="$(
    grep -RInE \
        --exclude-dir=.git \
        --exclude='*.docx' \
        --exclude='*.pdf' \
        --exclude='*.ipynb' \
        '(hf_[A-Za-z0-9]{20,}|ms-[A-Za-z0-9-]{20,}|modelscope[[:space:]]+login[[:space:]]+--token|password[[:space:]]*=|token[[:space:]]*=)' \
        "${SCRIPT_DIR}" || true
)"

if [[ -n "${SECRET_MATCHES}" ]]; then
    echo "[疑似密钥] 请人工检查以下内容："
    echo "${SECRET_MATCHES}"
    ERRORS=$((ERRORS + 1))
fi

if [[ ! -f "${SCRIPT_DIR}/download_assets.sh" ]]; then
    echo "[缺少文件] ${SCRIPT_DIR}/download_assets.sh"
    ERRORS=$((ERRORS + 1))
fi

if (( ERRORS > 0 )); then
    echo
    echo "预提交检查失败：发现 ${ERRORS} 个问题。"
    exit 1
fi

echo
echo "预提交检查通过：未发现权重、数据、日志、临时文件或超过 50 MiB 的文件。"
