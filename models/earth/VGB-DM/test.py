#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VGB-DM 完整流程测试脚本
基于之前成功修复的经验
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "disabled"

print("=" * 70)
print("VGB-DM 完整流程测试脚本")
print("=" * 70)
print(f"项目根目录: {PROJECT_ROOT}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"WANDB_MODE: {os.environ.get('WANDB_MODE')}")
print("=" * 70)


def run_command(cmd, description, check=False):
    """执行命令并打印输出"""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"  命令: {cmd}")
    print('-' * 70)
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"⚠️ 命令执行返回码: {result.returncode}")
    return result.returncode == 0


def find_files(pattern, path="."):
    """查找文件"""
    cmd = f"find {path} -name '{pattern}' 2>/dev/null"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    files = result.stdout.strip().split('\n')
    return [f for f in files if f]


def fix_lorenz_generator():
    """修复 Lorenz 数据生成脚本，使其使用 DCU/GPU"""
    generator_path = "src/data/generate_dataset/lorenz_attractor/generate_dataset.py"
    
    if not os.path.exists(generator_path):
        print(f"⚠️ 文件不存在: {generator_path}")
        return False
    
    print(f"\n🔧 修复 {generator_path}，强制使用 GPU/DCU...")
    
    with open(generator_path, 'r') as f:
        content = f.read()
    
    # 检查是否已经修复
    if 'device = torch.device("cuda")' in content:
        print("✅ 已修复，跳过")
        return True
    
    # 备份
    backup_path = generator_path + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"  备份: {backup_path}")
    
    # 修复: 替换 device 设置
    # 方式1: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content = content.replace(
        'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
        'device = torch.device("cuda")'
    )
    # 方式2: device = "cpu"
    content = content.replace(
        'device = "cpu"',
        'device = "cuda"'
    )
    # 方式3: torch.device("cpu")
    content = content.replace(
        'torch.device("cpu")',
        'torch.device("cuda")'
    )
    
    with open(generator_path, 'w') as f:
        f.write(content)
    
    print("✅ 修复完成！")
    return True


def create_wandb_util():
    """创建 wandb 工具文件"""
    wandb_path = "src/utils/wandb.py"
    
    if os.path.exists(wandb_path):
        print(f"✅ {wandb_path} 已存在")
        return True
    
    print(f"\n📝 创建 {wandb_path}...")
    with open(wandb_path, 'w') as f:
        f.write('''
"""
Weights & Biases utilities for VGB-DM.
"""
import wandb

class wandb_util:
    """Wrapper for wandb utility functions."""
    
    @staticmethod
    def init_wandb(*args, **kwargs):
        """Initialize wandb run and return config."""
        config = kwargs.get('config', {})
        project = kwargs.get('project', 'VGB-DM')
        name = kwargs.get('name', None)
        mode = kwargs.get('mode', 'offline')
        if mode != 'disabled':
            wandb.init(project=project, name=name, config=config, mode=mode)
        return config
    
    @staticmethod
    def log_metrics(metrics, step=None, **kwargs):
        """Log metrics to wandb."""
        wandb.log(metrics, step=step)
    
    @staticmethod
    def finish_wandb():
        """Finish wandb run."""
        wandb.finish()
    
    @staticmethod
    def watch(model, **kwargs):
        """Watch model gradients and parameters."""
        wandb.watch(model, **kwargs)
''')
    print(f"✅ {wandb_path} 创建完成")
    return True


def fix_base_encoder():
    """修复 base_encoder.py 中的 torch.types.Tensor 问题"""
    encoder_path = "src/models/encoders/base_encoder.py"
    
    if not os.path.exists(encoder_path):
        print(f"⚠️ 文件不存在: {encoder_path}")
        return False
    
    print(f"\n🔧 修复 {encoder_path}...")
    
    with open(encoder_path, 'r') as f:
        content = f.read()
    
    # 检查是否已修复
    if 'Tensor = torch.Tensor' in content:
        print("✅ 已修复，跳过")
        return True
    
    # 备份
    backup_path = encoder_path + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"  备份: {backup_path}")
    
    # 修复: 替换导入
    content = content.replace(
        'from torch.types import Tensor',
        'import torch\nTensor = torch.Tensor'
    )
    # 修复: Optional[Tensor] -> Optional[torch.Tensor]
    content = content.replace(
        'Optional[Tensor]',
        'Optional[torch.Tensor]'
    )
    # 修复: List[Tensor] -> List[torch.Tensor]
    content = content.replace(
        'List[Tensor]',
        'List[torch.Tensor]'
    )
    
    with open(encoder_path, 'w') as f:
        f.write(content)
    
    print("✅ 修复完成！")
    return True


def check_data_exists():
    """检查数据文件是否存在"""
    lorenz_files = find_files("*.pt", "experiments/dataset/lorenz_attractor")
    
    tr_file = None
    val_file = None
    test_file = None
    
    for f in lorenz_files:
        if "seed_42" in f or "tr_lorenz" in f:
            tr_file = f
        elif "seed_43" in f or "val_lorenz" in f:
            val_file = f
        elif "seed_44" in f or "test_lorenz" in f:
            test_file = f
        elif "seed_1" in f and "tr" in f:
            tr_file = f
        elif "seed_2" in f and "val" in f:
            val_file = f
        elif "seed_13" in f and "test" in f:
            test_file = f
    
    return tr_file, val_file, test_file


# ============================================================
# 第一步：修复代码
# ============================================================
print("\n" + "=" * 70)
print("第一步: 代码修复")
print("=" * 70)

# 1.1 修复 Lorenz 生成器
fix_lorenz_generator()

# 1.2 创建 wandb 工具
create_wandb_util()

# 1.3 修复 base_encoder
fix_base_encoder()


# ============================================================
# 第二步：检查环境
# ============================================================
print("\n" + "=" * 70)
print("第二步: 检查环境")
print("=" * 70)

# 检查 torch
result = subprocess.run("python -c 'import torch; print(torch.cuda.is_available())'", shell=True, capture_output=True, text=True)
if "True" in result.stdout:
    print("✅ GPU/DCU 可用")
else:
    print("⚠️ GPU/DCU 不可用，将使用 CPU")


# ============================================================
# 第三步：数据生成（使用修复后的脚本）
# ============================================================
print("\n" + "=" * 70)
print("第三步: 数据生成")
print("=" * 70)

# 检查是否已有数据
tr_file, val_file, test_file = check_data_exists()
print(f"\n当前数据文件:")
print(f"  训练集: {tr_file if tr_file else '未找到'}")
print(f"  验证集: {val_file if val_file else '未找到'}")
print(f"  测试集: {test_file if test_file else '未找到'}")

if tr_file and val_file and test_file:
    print("\n✅ 所有数据文件已存在，跳过生成")
else:
    print("\n生成 Lorenz 数据...")
    for seed in [42, 43, 44]:
        run_command(
            f"CUDA_VISIBLE_DEVICES=0 python src/data/generate_dataset/lorenz_attractor/generate_dataset.py --seed={seed} --n_trajectories=1000",
            f"Lorenz 数据生成 (seed={seed})"
        )


# ============================================================
# 第四步：Lorenz 模型训练
# ============================================================
print("\n" + "=" * 70)
print("第四步: Lorenz 模型训练")
print("=" * 70)

# 重新获取数据路径
tr_file, val_file, test_file = check_data_exists()

if tr_file and val_file:
    # 先移动文件到标准目录（如果需要）
    os.makedirs("experiments/dataset/lorenz_attractor/train", exist_ok=True)
    os.makedirs("experiments/dataset/lorenz_attractor/val", exist_ok=True)
    os.makedirs("experiments/dataset/lorenz_attractor/test", exist_ok=True)
    
    for f in find_files("*.pt", "experiments/dataset/lorenz_attractor"):
        if f and "train" not in f and "val" not in f and "test" not in f:
            if "seed_42" in f or "tr_lorenz" in f:
                dest = "experiments/dataset/lorenz_attractor/train/lorenz_data_seed_42.pt"
                if f != dest:
                    print(f"移动: {f} -> {dest}")
                    os.rename(f, dest)
                    tr_file = dest
            elif "seed_43" in f or "val_lorenz" in f:
                dest = "experiments/dataset/lorenz_attractor/val/lorenz_data_seed_43.pt"
                if f != dest:
                    print(f"移动: {f} -> {dest}")
                    os.rename(f, dest)
                    val_file = dest
            elif "seed_44" in f or "test_lorenz" in f:
                dest = "experiments/dataset/lorenz_attractor/test/lorenz_data_seed_44.pt"
                if f != dest:
                    print(f"移动: {f} -> {dest}")
                    os.rename(f, dest)
                    test_file = dest
    
    # 重新获取
    tr_file, val_file, test_file = check_data_exists()
    
    print(f"\n使用数据:")
    print(f"  训练集: {tr_file}")
    print(f"  验证集: {val_file}")
    print(f"  测试集: {test_file}")
    
    # 训练命令（快速测试）
    train_cmd = (
        f"CUDA_VISIBLE_DEVICES=0 python src/runner/run_hydra.py --multirun "
        f"exp=lorenz "
        f"exp.dataset.data_path_tr=\"./{tr_file}\" "
        f"exp.dataset.data_path_va=\"./{val_file}\" "
        f"exp.algorithm=dyn-fm "
        f"exp.sampler.sampler_mode=pairs-history "
        f"exp.sampler.max_length=-1 "
        f"exp.sampler.enc_len_episode=30 "
        f"exp.enc_model.len_episode=30 "
        f"exp.enc_model.p_dim=2 "
        f"exp.enc_model.z_dim=8 "
        f"exp.vf_model.phys_model=true "
        f"exp.vf_model.vf_phys=true "
        f"exp.vf_model.interpolation=linear "
        f"exp.vf_model.history_size=4 "
        f"exp.training.early_stopping=false "
        f"exp.optimization.vf_lr=1e-3 "
        f"exp.optimization.enc_lr=1e-4 "
        f"exp.optimization.beta_p=0.1 "
        f"exp.optimization.beta_z=0.1 "
        f"exp.optimization.n_epochs=3500 "
        f"exp.training.max_time_tr=120 "
        f"exp.seed=3"
    )
    
    print("\n开始训练 Lorenz 模型（3500 epochs，约 1-2 小时）...")
    print("提示: 按 Ctrl+C 可中断训练")
    run_command(train_cmd, "Lorenz 模型训练", check=False)
else:
    print("⚠️ 未找到完整的 Lorenz 数据集，跳过训练")


# ============================================================
# 第五部分：Lorenz 模型评估
# ============================================================
print("\n" + "=" * 70)
print("第五部分: Lorenz 模型评估")
print("=" * 70)

# 查找训练好的模型
checkpoints = find_files("best_chkpt.pth", "outputs/lorenz")
checkpoints = [c for c in checkpoints if c]

if checkpoints:
    best_chkpt = checkpoints[-1]
    root_dir = os.path.dirname(best_chkpt)
    print(f"找到检查点: {best_chkpt}")
    print(f"根目录: {root_dir}")
    
    if test_file:
        eval_cmd = f"python src/evaluate/evaluate.py --root_dir={root_dir} --dataset_path=./{test_file}"
        run_command(eval_cmd, "Lorenz 模型评估", check=False)
    else:
        print("⚠️ 未找到测试集，跳过评估")
else:
    print("⚠️ 未找到训练好的模型，跳过评估")


# ============================================================
# 第六部分：气候模型
# ============================================================
print("\n" + "=" * 70)
print("第六部分: 气候模型")
print("=" * 70)

if os.path.exists("experiments/dataset/era5_data"):
    print("✅ 找到 ERA5 数据目录")
    
    print("\n加载气候预训练模型...")
    run_command(
        "CUDA_VISIBLE_DEVICES=0 python -c \"from src.grey_box_clim.fm_phys_func import Climate_GBDM_ENC as GBDMModel; model = GBDMModel.from_pretrained('GurjeetSinghSangra/climate_GB_FM'); print('✅ 气候模型加载成功！')\"",
        "加载气候预训练模型",
        check=False
    )
else:
    print("❌ 未找到 ERA5 数据目录 (experiments/dataset/era5_data)")
    print("\n气候模型需要 ERA5 数据，当前跳过")
    print("如需运行，请从 WeatherBench 下载数据到 experiments/dataset/era5_data/")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("✅ 脚本执行完成！")
print("=" * 70)
print("\n📋 执行内容:")
print("   [1] 代码修复: Lorenz 生成器、wandb、base_encoder")
print("   [2] Lorenz: 数据生成 (seed=42,43,44)")
print("   [3] Lorenz: 模型训练 (3500 epochs)")
print("   [4] Lorenz: 模型评估")
print("   [5] 气候模型: 检查 ERA5 数据")
print("\n" + "=" * 70)