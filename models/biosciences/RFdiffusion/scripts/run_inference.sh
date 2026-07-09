#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"
python scripts/run_inference.py \
  'contigmap.contigs=[80-80]' \
  diffuser.T=15 \
  inference.final_step=15 \
  inference.num_designs=1 \
  inference.write_trajectory=False \
  inference.output_prefix=outputs/smoke/design
