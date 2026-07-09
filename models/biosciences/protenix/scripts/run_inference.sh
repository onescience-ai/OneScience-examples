#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export DATA_ROOT_DIR="${DATA_ROOT_DIR:-../bio_protenix_dataset}"
python3 scripts/run_inference.py "$@"
