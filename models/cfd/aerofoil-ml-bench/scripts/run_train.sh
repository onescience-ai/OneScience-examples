#!/bin/bash
#SBATCH -p hx1hdnormal01
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH -J aerofoil_train
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=3GB
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

module purge
source /etc/profile
source /etc/profile.d/modules.sh
module use /work2/share/sghpc_sdk/modulefiles/
module load sghpcdas/25.6
module load compiler/dtk/25.04.4
module load sghpc-mpi-gcc/26.3
source /work2/share/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
conda activate onescience311
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

cd /public/home/wangqi_scnet/auto_run_test/repro
DATA_DIR=/public/share/sugonhpcapp01/onestore/onedatasets/CFD_Benchmark/airfoil

echo "=== Training model: $MODEL ==="
python main.py "$MODEL" \
  --data-dir "$DATA_DIR" \
  --out-dir metrics \
  --foils "$NFOILS" \
  --epochs "$NEPOCHS" \
  --subsample "$SUBSAMPLE" \
  --n-total 120 \
  --n-val 30 \
  --n-test 30 \
  --seed 42 \
  --graph
echo "TRAIN_DONE model=$MODEL"
