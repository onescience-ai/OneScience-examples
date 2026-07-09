#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${RF_DIFFUSION_MODELSCOPE_MODEL:-OneScience/RFdiffusion}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST_FILE:-$SCRIPT_DIR/MODEL_FILE_MANIFEST.tsv}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    exit 1
fi

"$PYTHON_BIN" -m pip install modelscope

if ! command -v modelscope >/dev/null 2>&1; then
    echo "modelscope command not found after installation. Please check your Python environment PATH." >&2
    exit 1
fi

download_asset() {
    local remote_path="$1"
    local target_path="$SCRIPT_DIR/$remote_path"
    local target_dir
    local flat_path

    target_dir="$(dirname "$target_path")"
    flat_path="$SCRIPT_DIR/$(basename "$remote_path")"

    mkdir -p "$target_dir"
    echo "Downloading $remote_path"
    modelscope download --model "$MODEL_ID" "$remote_path" --local_dir "$SCRIPT_DIR"

    if [[ -f "$target_path" ]]; then
        return 0
    fi

    if [[ -f "$flat_path" ]]; then
        mv -f "$flat_path" "$target_path"
        return 0
    fi

    echo "Download completed but expected file was not found: $target_path" >&2
    exit 1
}

ASSET_PATHS=(
    "examples/input_pdbs/1qys.pdb"
    "examples/input_pdbs/1YCR.pdb"
)

while IFS=$'\t' read -r _name _size _sha256 package_path; do
    package_path="${package_path%$'\r'}"

    if [[ "$package_path" == "package_path" || -z "$package_path" ]]; then
        continue
    fi

    case "$package_path" in
        weight/*)
            ASSET_PATHS+=("$package_path")
            ;;
    esac
done < "$MANIFEST"

for asset_path in "${ASSET_PATHS[@]}"; do
    download_asset "$asset_path"
done

echo "RFdiffusion examples and weights are ready under $SCRIPT_DIR/examples and $SCRIPT_DIR/weight."
