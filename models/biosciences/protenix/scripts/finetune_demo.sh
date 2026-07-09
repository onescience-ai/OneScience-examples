#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export DATA_ROOT_DIR="${DATA_ROOT_DIR:-../bio_protenix_dataset}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
python3 scripts/finetune.py "$@"
