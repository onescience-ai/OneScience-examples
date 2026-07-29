#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://huggingface.co/lschmidt/edsr-dsc/resolve/main"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$SCRIPT_DIR}"

download_and_verify() {
    local relative_path="$1"
    local expected_sha256="$2"
    local output_path="${OUTPUT_DIR}/${relative_path}"
    local part_path="${output_path}.part"
    local actual_sha256

    mkdir -p "$(dirname "$output_path")"

    if [ -f "$output_path" ]; then
        actual_sha256="$(sha256sum "$output_path" | awk '{print $1}')"
        if [ "$actual_sha256" = "$expected_sha256" ]; then
            echo "Already present and verified: $output_path"
            return 0
        fi

        echo "ERROR: Existing file has an unexpected SHA256: $output_path" >&2
        echo "Expected: $expected_sha256" >&2
        echo "Actual:   $actual_sha256" >&2
        echo "Refusing to overwrite the existing file." >&2
        return 1
    fi

    echo "Downloading: ${BASE_URL}/${relative_path}?download=true"

    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 5 --retry-delay 5 \
            --continue-at - \
            --output "$part_path" \
            "${BASE_URL}/${relative_path}?download=true"
    elif command -v wget >/dev/null 2>&1; then
        wget --continue \
            --output-document="$part_path" \
            "${BASE_URL}/${relative_path}?download=true"
    else
        echo "ERROR: curl or wget is required." >&2
        return 1
    fi

    actual_sha256="$(sha256sum "$part_path" | awk '{print $1}')"
    if [ "$actual_sha256" != "$expected_sha256" ]; then
        echo "ERROR: SHA256 verification failed for $relative_path" >&2
        echo "Expected: $expected_sha256" >&2
        echo "Actual:   $actual_sha256" >&2
        echo "Incomplete file retained at: $part_path" >&2
        return 1
    fi

    mv "$part_path" "$output_path"
    echo "Downloaded and verified: $output_path"
}

download_and_verify \
    "pytorch_model_4x.pt" \
    "60a0798fdd2b001ce82b3065b25e08d8179b346e96cef287c2129107cdb28d51"

download_and_verify \
    "test_data/test_wind_velocities.nc" \
    "f47ecd109deb982cbbbde592e8faf9fabb1ef8086ed4b881ed785e2b3d5770a5"

echo "All EDSR-DSC resources are ready."
