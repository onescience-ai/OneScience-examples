#!/bin/bash
#SBATCH -p largedev
#SBATCH -N 4
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH -J FourCastNet
#SBATCH --time=72:00:00
#SBATCH -o logs/%j.out
#SBATCH --exclusive

export OMP_NUM_THREADS=16
export MASTER_ADDR=$(hostname)

cd "$(dirname "$0")/.."
mkdir -p logs

echo SLURM_NNODES=$SLURM_NNODES
echo SLURM_NTASKS=$SLURM_NTASKS

# 按集群环境需要加载 PyTorch/DCU/GPU 运行环境，例如：
# source /path/to/env.sh

srun -u --mpi=pmix\
    bash -c "
    python scripts/train.py
    "
