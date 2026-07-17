#!/bin/bash
#SBATCH --job-name=dp_pt_atten
#SBATCH --partition=hx1hdexclu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

# 统一走 matchem_env.sh 路径（加载模块、激活 conda、设置基础变量）
source "$SCRIPT_DIR/../../matchem_env.sh"

# DeepMD 训练环境已由 matchem_env.sh 覆盖

# 单卡训练
cd "$SCRIPT_DIR"
dp --pt train input_torch.json
