from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from convgru_ensemble import RadarLightningModel


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "examples" / "sample_small_6x64.nc"
CHECKPOINT = ROOT / "model.ckpt"
OUTPUT = ROOT / "predictions_smoke.nc"

VARIABLE = "RR"
FORECAST_STEPS = 1
ENSEMBLE_SIZE = 1
DEVICE = "cpu"


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"找不到小数据文件: {INPUT}\n请先运行 02_make_small_input.py"
        )
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"找不到模型权重: {CHECKPOINT}")

    print("=" * 72)
    print("IRENE 小数据推理测试")
    print("=" * 72)
    print(f"设备: {DEVICE}")
    print(f"输入: {INPUT}")
    print(f"权重: {CHECKPOINT}")
    print(f"预测步数: {FORECAST_STEPS}")
    print(f"集合成员数: {ENSEMBLE_SIZE}")
    print()

    with xr.open_dataset(INPUT) as ds:
        if VARIABLE not in ds:
            raise KeyError(f"输入文件中没有变量 {VARIABLE}，现有变量: {list(ds.data_vars)}")

        radar = ds[VARIABLE].load()
        if radar.ndim != 3:
            raise ValueError(
                f"模型输入必须为三维 (T,H,W)，当前 dims={radar.dims}, shape={radar.shape}"
            )

        time_dim = "time" if "time" in radar.dims else radar.dims[0]
        spatial_dims = [dim for dim in radar.dims if dim != time_dim]
        radar = radar.transpose(time_dim, spatial_dims[0], spatial_dims[1])
        past = np.asarray(radar.values, dtype=np.float32)

    if past.shape != (6, 64, 64):
        raise ValueError(f"本测试预期输入 (6,64,64)，实际为 {past.shape}")

    print(f"输入形状: {past.shape}")
    print(f"输入数据类型: {past.dtype}")
    print(f"输入 NaN 数量: {int(np.isnan(past).sum())}")

    load_start = time.perf_counter()
    print("\n正在加载 checkpoint，请耐心等待……")
    model = RadarLightningModel.from_checkpoint(
        checkpoint_path=str(CHECKPOINT),
        device=DEVICE,
    )
    load_seconds = time.perf_counter() - load_start
    print(f"模型加载完成，用时 {load_seconds:.2f} 秒")
    print(f"模型参数所在设备: {model.device}")

    predict_start = time.perf_counter()
    print("\n正在执行一次小规模前向预测……")
    predictions = model.predict(
        past,
        forecast_steps=FORECAST_STEPS,
        ensemble_size=ENSEMBLE_SIZE,
    )
    predict_seconds = time.perf_counter() - predict_start

    predictions = np.asarray(predictions, dtype=np.float32)
    expected_shape = (
        ENSEMBLE_SIZE,
        FORECAST_STEPS,
        past.shape[1],
        past.shape[2],
    )

    print(f"预测完成，用时 {predict_seconds:.2f} 秒")
    print(f"预测形状: {predictions.shape}")
    print(f"预期形状: {expected_shape}")

    if predictions.shape != expected_shape:
        raise RuntimeError(
            f"预测形状不符合预期：实际 {predictions.shape}，预期 {expected_shape}"
        )

    finite_ratio = float(np.isfinite(predictions).mean())
    print(f"有限数值比例: {finite_ratio:.6f}")
    print(f"预测最小值: {float(np.nanmin(predictions)):.6f}")
    print(f"预测最大值: {float(np.nanmax(predictions)):.6f}")
    print(f"预测平均值: {float(np.nanmean(predictions)):.6f}")

    if finite_ratio < 1.0:
        raise RuntimeError("预测中包含 NaN 或无穷值。")

    output = xr.Dataset(
        data_vars={
            "precipitation_forecast": (
                (
                    "ensemble_member",
                    "forecast_step",
                    spatial_dims[0],
                    spatial_dims[1],
                ),
                predictions,
            )
        },
        coords={
            "ensemble_member": np.arange(ENSEMBLE_SIZE, dtype=np.int32),
            "forecast_step": np.arange(1, FORECAST_STEPS + 1, dtype=np.int32),
        },
        attrs={
            "model": "IRENE",
            "checkpoint": str(CHECKPOINT),
            "input_file": str(INPUT),
            "device": DEVICE,
            "forecast_steps": FORECAST_STEPS,
            "ensemble_size": ENSEMBLE_SIZE,
            "checkpoint_load_seconds": load_seconds,
            "prediction_seconds": predict_seconds,
            "units": "mm/h",
        },
    )

    for dim in spatial_dims:
        if dim in radar.coords and radar.coords[dim].ndim == 1:
            output = output.assign_coords({dim: radar.coords[dim].values})

    output.to_netcdf(OUTPUT)

    print()
    print("=" * 72)
    print("推理测试成功")
    print(f"结果文件: {OUTPUT}")
    print("=" * 72)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
