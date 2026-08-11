#!/usr/bin/env bash
set -euo pipefail

FILE_NAME="aifs-single-mse-1.0.ckpt"
URL="https://huggingface.co/ecmwf/aifs-single-1.0/resolve/main/${FILE_NAME}?download=true"
EXPECTED_BYTES="994084883"
EXPECTED_SHA256="1fed399c097c0127d5bbe074f4f8bbc123759736145d990699c215ff07543ccd"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:-${SCRIPT_DIR}/${FILE_NAME}}"
PART_FILE="${OUTPUT}.part"

verify_checkpoint() {
    local path="$1"
    local actual_bytes
    local actual_sha256

    actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
    if [ "$actual_bytes" != "$EXPECTED_BYTES" ]; then
        echo "ERROR: Unexpected checkpoint size: $actual_bytes bytes" >&2
        echo "Expected: $EXPECTED_BYTES bytes" >&2
        return 1
    fi

    actual_sha256="$(sha256sum "$path" | awk '{print $1}')"
    if [ "$actual_sha256" != "$EXPECTED_SHA256" ]; then
        echo "ERROR: Checkpoint SHA256 verification failed." >&2
        echo "Expected: $EXPECTED_SHA256" >&2
        echo "Actual:   $actual_sha256" >&2
        return 1
    fi
}

mkdir -p "$(dirname "$OUTPUT")"

if [ -f "$OUTPUT" ]; then
    if verify_checkpoint "$OUTPUT"; then
        echo "Checkpoint already exists; size and SHA256 are correct:"
        echo "$OUTPUT"
        exit 0
    fi

    echo "Refusing to overwrite the existing checkpoint: $OUTPUT" >&2
    exit 1
fi

echo "Downloading official AIFS checkpoint..."
echo "Source:      $URL"
echo "Destination: $OUTPUT"

if command -v curl >/dev/null 2>&1; then
    CURL_ARGS=(
        -fL
        --retry 5
        --retry-delay 5
        --continue-at -
        --output "$PART_FILE"
    )
    if [ -n "${HF_TOKEN:-}" ]; then
        CURL_ARGS+=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi
    curl "${CURL_ARGS[@]}" "$URL"
elif command -v wget >/dev/null 2>&1; then
    WGET_ARGS=(--continue --output-document="$PART_FILE")
    if [ -n "${HF_TOKEN:-}" ]; then
        WGET_ARGS+=(--header="Authorization: Bearer ${HF_TOKEN}")
    fi
    wget "${WGET_ARGS[@]}" "$URL"
else
    echo "ERROR: curl or wget is required." >&2
    exit 1
fi

if ! verify_checkpoint "$PART_FILE"; then
    echo "Incomplete file retained at: $PART_FILE" >&2
    exit 1
fi

mv "$PART_FILE" "$OUTPUT"

echo "Download completed; size and SHA256 verified:"
echo "$OUTPUT"
