from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parent
SMALL_INPUT = ROOT / "examples" / "sample_small_6x64.nc"
PREDICTION = ROOT / "predictions_smoke.nc"
INPUT_IMAGE = ROOT / "smoke_input_last_frame.png"
FORECAST_IMAGE = ROOT / "smoke_forecast_first_step.png"
REPORT = ROOT / "smoke_test_report.txt"


def save_field(field: np.ndarray, title: str, output: Path) -> None:
    plt.figure(figsize=(6, 5))
    image = plt.imshow(field)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(image, label="Rain rate (mm/h)")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    if not SMALL_INPUT.exists():
        raise FileNotFoundError(f"找不到输入文件: {SMALL_INPUT}")
    if not PREDICTION.exists():
        raise FileNotFoundError(
            f"找不到预测文件: {PREDICTION}\n请先运行 03_run_smoke_test.py"
        )

    with xr.open_dataset(SMALL_INPUT) as input_ds:
        input_values = np.asarray(input_ds["RR"].values)
        last_input = input_values[-1]

    with xr.open_dataset(PREDICTION) as prediction_ds:
        if "precipitation_forecast" not in prediction_ds:
            raise KeyError(
                "预测文件中没有 precipitation_forecast 变量。"
            )
        forecast_values = np.asarray(
            prediction_ds["precipitation_forecast"].values
        )
        first_forecast = forecast_values[0, 0]
        dataset_text = str(prediction_ds)

    finite_ratio = float(np.isfinite(forecast_values).mean())

    lines = [
        "IRENE smoke test report",
        "=" * 60,
        f"Small input file: {SMALL_INPUT}",
        f"Prediction file: {PREDICTION}",
        f"Input shape: {input_values.shape}",
        f"Prediction shape: {forecast_values.shape}",
        f"Prediction finite ratio: {finite_ratio:.6f}",
        f"Prediction min: {float(np.nanmin(forecast_values)):.6f}",
        f"Prediction max: {float(np.nanmax(forecast_values)):.6f}",
        f"Prediction mean: {float(np.nanmean(forecast_values)):.6f}",
        "",
        "Prediction dataset:",
        dataset_text,
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    save_field(last_input, "IRENE: last observed input frame", INPUT_IMAGE)
    save_field(first_forecast, "IRENE: first forecast step", FORECAST_IMAGE)

    print("\n".join(lines[:9]))
    print()
    print(f"输入图像: {INPUT_IMAGE}")
    print(f"预测图像: {FORECAST_IMAGE}")
    print(f"文本报告: {REPORT}")

    if forecast_values.shape != (1, 1, 64, 64):
        raise RuntimeError(
            f"预测形状不是预期的 (1,1,64,64)，实际为 {forecast_values.shape}"
        )
    if finite_ratio != 1.0:
        raise RuntimeError("预测结果包含 NaN 或无穷值。")

    print("\n检查通过：小数据主推理流程已完整跑通。")


if __name__ == "__main__":
    main()
