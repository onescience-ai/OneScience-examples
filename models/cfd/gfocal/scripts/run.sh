#!/bin/bash
# Training script for gfocal on AirfRANS dataset
# Usage: bash run.sh

MODEL="gfocal"
DATA_DIR="./data/airfrans/data/Dataset"
OUTPUT_DIR="./output"

python3 train.py \
  --model "$gfocal" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 20 \
  --sample-points 4000 \
  --device cpu
