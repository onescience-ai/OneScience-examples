#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs

echo "============================================================"
echo "MetNet environment check"
echo "============================================================"

python - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("Accelerator available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Accelerator:", torch.cuda.get_device_name(0))
else:
    print("Accelerator: CPU")
PY

if [ ! -f "hf_snapshot/pytorch_model.bin" ]; then
    echo
    echo "MetNet weight was not found."
    echo "Downloading it from Hugging Face..."
    python download_weights.py
fi

echo
echo "============================================================"
echo "Running MetNet synthetic-input inference test"
echo "============================================================"

PYTHONPATH="$ROOT_DIR" \
python test_metnet_inference.py \
2>&1 | tee outputs/metnet_inference.log
