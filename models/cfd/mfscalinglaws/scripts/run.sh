#!/bin/bash
# Training script for mfscalinglaws on AirfRANS dataset
# Usage: bash run.sh

MODEL="mfscalinglaws"
DATA_DIR="./data/airfrans/data/Dataset"
OUTPUT_DIR="./output"

python3 train.py \
  --model "$mfscalinglaws" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 20 \
  --sample-points 4000 \
  --device cpu
