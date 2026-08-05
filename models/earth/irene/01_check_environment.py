from __future__ import annotations

import platform
import sys
from pathlib import Path

import xarray as xr


ROOT = Path(__file__).resolve().parent

REQUIRED_PATHS = [
    ROOT / "model.ckpt",
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
    ROOT / "examples" / "sample_data.nc",
    ROOT / "convgru_ensemble" / "__init__.py",
    ROOT / "convgru_ensemble" / "model.py",
    ROOT / "convgru_ensemble" / "lightning_model.py",
]


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def main() -> None:
    print("=" * 72)
    print("IRENE 环境与文件检查")
    print("=" * 72)
    print(f"项目根目录: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"解释器: {sys.executable}")
    print(f"操作系统: {platform.platform()}")
    print()

    missing: list[Path] = []
    print("[1] 检查关键文件")
    for path in REQUIRED_PATHS:
        if path.exists():
            suffix = f" ({human_size(path.stat().st_size)})" if path.is_file() else ""
            print(f"  [存在] {path.relative_to(ROOT)}{suffix}")
        else:
            missing.append(path)
            print(f"  [缺失] {path.relative_to(ROOT)}")

    if missing:
        raise FileNotFoundError("存在缺失文件，请先补齐后再继续。")

    print()
    print("[2] 检查关键 Python 包")
    try:
        import numpy as np
        print(f"  NumPy: {np.__version__}")
    except Exception as exc:
        raise RuntimeError(f"NumPy 导入失败: {exc}") from exc

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 设备: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        raise RuntimeError(f"PyTorch 导入失败: {exc}") from exc

    try:
        import pytorch_lightning as pl
        print(f"  PyTorch Lightning: {pl.__version__}")
    except Exception as exc:
        raise RuntimeError(f"PyTorch Lightning 导入失败: {exc}") from exc

    try:
        import matplotlib
        print(f"  Matplotlib: {matplotlib.__version__}")
    except Exception as exc:
        raise RuntimeError(f"Matplotlib 导入失败: {exc}") from exc

    try:
        from convgru_ensemble import RadarLightningModel
        print(f"  IRENE 模型类: {RadarLightningModel.__name__}")
    except Exception as exc:
        raise RuntimeError(f"IRENE 源代码导入失败: {exc}") from exc

    print()
    print("[3] 检查示例 NetCDF")
    sample_path = ROOT / "examples" / "sample_data.nc"
    with xr.open_dataset(sample_path) as ds:
        print(ds)
        if "RR" in ds.data_vars:
            variable = "RR"
        elif len(ds.data_vars) == 1:
            variable = next(iter(ds.data_vars))
            print(f"  未找到 RR，自动识别唯一变量: {variable}")
        else:
            raise KeyError(
                f"没有 RR 且包含多个变量，无法自动判断。变量为: {list(ds.data_vars)}"
            )

        radar = ds[variable]
        if radar.ndim != 3:
            raise ValueError(
                f"雷达变量必须是三维 (T,H,W)，当前 dims={radar.dims}, shape={radar.shape}"
            )
        print(f"  使用变量: {variable}")
        print(f"  维度: {radar.dims}")
        print(f"  形状: {radar.shape}")
        print(f"  数据类型: {radar.dtype}")

    model_size = (ROOT / "model.ckpt").stat().st_size
    if model_size < 100_000_000:
        raise RuntimeError(
            "model.ckpt 小于 100 MB，可能只是大文件指针或下载不完整。"
        )

    print()
    print("=" * 72)
    print("检查通过：可以继续制作小数据并运行推理测试。")
    print("=" * 72)


if __name__ == "__main__":
    main()
