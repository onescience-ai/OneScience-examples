#!/bin/bash
# GeoANF training on AirfRANS
# Execution channel: local_direct

module load sghpc-mpi-gcc/26.3 2>/dev/null
module load sghpcdas/25.6 2>/dev/null

cd /public/home/wangqi_scnet/batch_paper_repo/batch_output/airfrans/GeoANF

echo "Starting GeoANF training..."
echo "Data dir: ${AIRFRANS_DATA_DIR:-./data/airfrans}"
echo "Output dir: ./output"

python3 train.py \
  --data-dir /public/home/wangqi_scnet/batch_paper_repo/data/airfrans \
  --stats-dir /public/home/wangqi_scnet/batch_paper_repo/data/airfrans/stats \
  --output-dir ./output \
  --epochs 15 \
  --lr 0.003 \
  --sample-points 16000 \
  --loss-alpha 1.0 \
  --hidden-dim 64 \
  --num-heads 8 \
  --seed 42
