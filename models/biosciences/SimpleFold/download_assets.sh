#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"
MODELSCOPE_MODEL="${MODELSCOPE_MODEL:-OneScience/SimpleFold}"
MANIFEST_FILE="${MANIFEST_FILE:-$ROOT_DIR/MODEL_FILE_MANIFEST.tsv}"

echo "== SimpleFold assets download =="
echo "Package root: $ROOT_DIR"
echo "ModelScope model: $MODELSCOPE_MODEL"

$PIP_BIN install modelscope

if ! command -v modelscope >/dev/null 2>&1; then
  echo "ERROR: modelscope command was not found after installation." >&2
  echo "Please make sure the Python scripts directory is in PATH." >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/examples" "$ROOT_DIR/weight/esm_models"

ASSET_PATHS=("examples/minimal.fasta")

if [[ -f "$MANIFEST_FILE" ]]; then
  while IFS=$'\t' read -r _file _size _sha256 package_path; do
    package_path="${package_path%$'\r'}"
    if [[ "$package_path" == weight/* ]]; then
      ASSET_PATHS+=("$package_path")
    fi
  done < "$MANIFEST_FILE"
else
  echo "ERROR: manifest file not found: $MANIFEST_FILE" >&2
  exit 1
fi

for asset_path in "${ASSET_PATHS[@]}"; do
  echo "Downloading $asset_path"
  modelscope download --model "$MODELSCOPE_MODEL" "$asset_path" --local_dir "$ROOT_DIR"
done

echo "Assets downloaded under examples/ and weight/."
