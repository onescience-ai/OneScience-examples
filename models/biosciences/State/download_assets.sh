#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELSCOPE_MODEL="${MODELSCOPE_MODEL:-OneScience/State}"

echo "== State assets download =="
echo "Package root: ${ROOT_DIR}"
echo "ModelScope model: ${MODELSCOPE_MODEL}"

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

mkdir -p "${ROOT_DIR}/weights" "${ROOT_DIR}/examples"

modelscope download \
    --model "${MODELSCOPE_MODEL}" \
    --include "weights/**" "examples/**" \
    --local_dir "${ROOT_DIR}"

missing_assets=()
for asset_dir in weights examples; do
    if [[ ! -d "${ROOT_DIR}/${asset_dir}" ]] || \
        [[ -z "$(find "${ROOT_DIR}/${asset_dir}" -type f -print -quit)" ]]; then
        missing_assets+=("${asset_dir}/")
    fi
done

if (( ${#missing_assets[@]} > 0 )); then
    echo "ERROR: The following asset directories are missing or empty after download:" >&2
    printf '  - %s\n' "${missing_assets[@]}" >&2
    exit 1
fi

echo "State assets are ready under ${ROOT_DIR}/weights and ${ROOT_DIR}/examples."
