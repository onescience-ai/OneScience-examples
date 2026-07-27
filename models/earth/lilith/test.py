#!/usr/bin/env python3
"""
LILITH Complete Training Script
一键训练LILITH模型（30轮）

直接调用 training.train_simple 模块，使用当前Python环境

Usage:
    python test.py
    python test.py --epochs 30 --batch-size 64 --d-model 128 --layers 4
"""

import os
import sys
import argparse
import importlib.util
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================
# 配置
# ============================================================

CONFIG = {
    "data_dir": "data/processed/training",
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
}

# ============================================================
# 工具函数
# ============================================================

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_step(text):
    print(f"\n▶ {text}...")

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_warning(text):
    print(f"⚠️  {text}")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def file_exists(path):
    return os.path.exists(path)

# ============================================================
# 步骤1: 数据准备
# ============================================================

def prepare_data():
    """准备训练数据"""
    print_header("步骤 1/2: 准备数据")
    
    ensure_dir(CONFIG["data_dir"])
    
    # 检查数据是否已存在
    X_path = os.path.join(CONFIG["data_dir"], "X.npy")
    Y_path = os.path.join(CONFIG["data_dir"], "Y.npy")
    
    if file_exists(X_path) and file_exists(Y_path):
        print_success(f"数据已存在: {CONFIG['data_dir']}")
        return True
    
    print_warning("未找到训练数据，将使用合成数据")
    generate_synthetic_data()
    return True

def generate_synthetic_data():
    """生成合成数据（测试用）"""
    import numpy as np
    
    print_step("生成合成数据")
    
    n_samples = 10000
    seq_len = 90
    n_features = 3
    
    t = np.linspace(0, 4*np.pi, seq_len)
    
    X = np.zeros((n_samples, seq_len, n_features))
    Y = np.zeros((n_samples, seq_len, n_features))
    
    for i in range(n_samples):
        amp_temp = 10 + np.random.randn() * 2
        phase_temp = np.random.randn() * 0.5
        temp = amp_temp * np.sin(t + phase_temp) + 15 + np.random.randn(seq_len) * 2
        
        precip = np.random.exponential(2, seq_len) * (np.sin(t) + 1) / 2
        pressure = 1013 + 10 * np.sin(t/2 + np.random.randn()) + np.random.randn(seq_len) * 5
        
        X[i, :, 0] = temp
        X[i, :, 1] = precip
        X[i, :, 2] = pressure
        
        Y[i, :, :] = np.roll(X[i, :, :], shift=-1, axis=0)
        Y[i, -1, :] = X[i, -1, :]
    
    np.save(os.path.join(CONFIG["data_dir"], "X.npy"), X)
    np.save(os.path.join(CONFIG["data_dir"], "Y.npy"), Y)
    
    print_success(f"生成 {n_samples} 条合成数据")

# ============================================================
# 步骤2: 训练模型（使用subprocess调用，但使用当前Python）
# ============================================================

def train_model(args):
    """执行训练 - 使用subprocess调用，使用当前Python"""
    print_header("步骤 2/2: 训练模型")
    
    ensure_dir(CONFIG["checkpoint_dir"])
    ensure_dir(CONFIG["log_dir"])
    
    print_step("执行 README 中的 Quick Training (30轮)")
    
    # 使用当前Python路径
    python_path = sys.executable
    print(f"  使用Python: {python_path}")
    
    # 检查当前Python是否能识别DCU
    try:
        import torch
        print(f"  PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  🚀 DCU: ✅ ({torch.cuda.get_device_name(0)})")
        else:
            print("  ⚠️ 当前Python无法识别DCU，尝试设置环境变量...")
    except ImportError:
        print_warning("PyTorch未安装")
    
    # 构建命令
    cmd = [
        python_path,
        "-m", "training.train_simple",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--d-model", str(args.d_model),
        "--layers", str(args.layers),
        "--lr", str(args.lr),
    ]
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    # 打印命令
    cmd_str = " ".join(cmd)
    print(f"\n  运行命令:")
    print(f"  $ {cmd_str}")
    print()
    
    try:
        # 构建环境变量 - 强制使用DCU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["HIP_VISIBLE_DEVICES"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # 使用subprocess运行
        result = subprocess.run(
            cmd,
            env=env,
            check=False
        )
        
        if result.returncode == 0:
            print_success("训练完成")
            return True
        else:
            print_error(f"训练失败 (退出码: {result.returncode})")
            return False
            
    except Exception as e:
        print_error(f"训练失败: {e}")
        return False

# ============================================================
# 步骤3: 验证模型
# ============================================================

def verify_model():
    """验证训练好的模型"""
    best_model_path = os.path.join(CONFIG["checkpoint_dir"], "lilith_best.pt")
    
    if not file_exists(best_model_path):
        print_warning("未找到训练好的模型")
        return False
    
    try:
        import torch
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        print(f"  模型文件: {best_model_path}")
        print(f"  模型大小: {os.path.getsize(best_model_path) / 1024 / 1024:.2f} MB")
        
        if 'val_loss' in checkpoint:
            print(f"  验证损失: {checkpoint['val_loss']:.4f}")
            rmse = checkpoint['val_loss'] ** 0.5
            print(f"  RMSE: {rmse:.4f}°C")
        
        if 'config' in checkpoint:
            print(f"  模型配置: {checkpoint['config']}")
        
        print_success("模型验证通过")
        return True
    except Exception as e:
        print_error(f"验证失败: {e}")
        return False

# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='LILITH 30轮训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test.py                              # 使用默认参数（30轮）
  python test.py --epochs 50                  # 训练50轮
  python test.py --batch-size 128             # 批次大小128
  python test.py --d-model 256 --layers 6    # 更大的模型
  python test.py --resume checkpoints/lilith_best.pt  # 从检查点恢复
        """
    )
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数 (默认: 30)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批次大小 (默认: 64)')
    parser.add_argument('--d-model', type=int, default=128,
                       help='模型维度 (默认: 128)')
    parser.add_argument('--layers', type=int, default=4,
                       help='Transformer层数 (默认: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率 (默认: 1e-4)')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print_header("LILITH Quick Training (30轮)")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  当前Python: {sys.executable}")
    
    # 检查当前Python是否能识别DCU
    try:
        import torch
        print(f"  PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  🚀 DCU: ✅ ({device_name})")
            print(f"  💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠️ DCU不可用，将使用CPU")
            print("  提示: 如需使用DCU，请确保使用正确的Python环境")
    except ImportError:
        print_warning("PyTorch未安装")
    
    print("\n训练参数 (与README Quick Training一致):")
    print(f"  --epochs {args.epochs}")
    print(f"  --batch-size {args.batch_size}")
    print(f"  --d-model {args.d_model}")
    print(f"  --layers {args.layers}")
    print(f"  --lr {args.lr}")
    if args.resume:
        print(f"  --resume {args.resume}")
    
    try:
        # 步骤1: 准备数据
        prepare_data()
        
        # 步骤2: 训练模型
        success = train_model(args)
        
        if success:
            # 步骤3: 验证模型
            verify_model()
            
            print_header("训练完成 🎉")
            print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  模型保存: {CONFIG['checkpoint_dir']}/lilith_best.pt")
            print("\n使用训练好的模型:")
            print(f"  python -m inference.forecast --checkpoint {CONFIG['checkpoint_dir']}/lilith_best.pt")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被中断")
        return 1
    except Exception as e:
        print_error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()