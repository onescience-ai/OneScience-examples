#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${PROTENIX_MODELSCOPE_MODEL:-OneScience/protenix}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="$SCRIPT_DIR/MODEL_FILE_MANIFEST.tsv"

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

while IFS=$'\t' read -r _name _size _sha256 package_path; do
    package_path="${package_path%$'\r'}"

    if [[ "$package_path" == "package_path" || -z "$package_path" ]]; then
        continue
    fi

    case "$package_path" in
        examples/*|weight/*)
            download_asset "$package_path"
            ;;
    esac
done < "$MANIFEST"

echo "Protenix examples and weights are ready under $SCRIPT_DIR/examples and $SCRIPT_DIR/weight."
