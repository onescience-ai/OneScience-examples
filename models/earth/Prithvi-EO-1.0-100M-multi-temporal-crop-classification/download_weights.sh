#!/usr/bin/env bash
set -euo pipefail

FILE_NAME="multi_temporal_crop_classification_Prithvi_100M.pth"
URL="https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification/resolve/main/${FILE_NAME}?download=true"
EXPECTED_SHA256="37ed41637eccccec65ca2031324e2c03a4f168e1ea0ea71ad180910589fa018c"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:-${SCRIPT_DIR}/${FILE_NAME}}"
PART_FILE="${OUTPUT}.part"

mkdir -p "$(dirname "$OUTPUT")"

if [ -f "$OUTPUT" ]; then
    ACTUAL_SHA256="$(sha256sum "$OUTPUT" | awk '{print $1}')"
    if [ "$ACTUAL_SHA256" = "$EXPECTED_SHA256" ]; then
        echo "Checkpoint already exists and SHA256 is correct:"
        echo "$OUTPUT"
        exit 0
    fi

    echo "ERROR: Existing file has an unexpected SHA256." >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    echo "Actual:   $ACTUAL_SHA256" >&2
    echo "Refusing to overwrite: $OUTPUT" >&2
    exit 1
fi

echo "Downloading official checkpoint..."
echo "Source:      $URL"
echo "Destination: $OUTPUT"

if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 5 \
        --continue-at - \
        --output "$PART_FILE" \
        "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget --continue --output-document="$PART_FILE" "$URL"
else
    echo "ERROR: curl or wget is required." >&2
    exit 1
fi

ACTUAL_SHA256="$(sha256sum "$PART_FILE" | awk '{print $1}')"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: Downloaded checkpoint failed SHA256 verification." >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    echo "Actual:   $ACTUAL_SHA256" >&2
    echo "Incomplete file retained at: $PART_FILE" >&2
    exit 1
fi

mv "$PART_FILE" "$OUTPUT"

echo "Download completed and SHA256 verified:"
echo "$OUTPUT"
