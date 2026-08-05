#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harley-ml/Hweh-6M 一键复现脚本
==============================

功能
----
1. 固定到 Hugging Face 提交 710d8df，下载模型代码、配置与 Safetensors 权重；
2. 默认自动生成 72 小时、22 特征的仿真天气序列；
3. 可选从 Open-Meteo 获取最近 72 小时真实历史天气；
4. 运行 Hweh-6M，预测未来 12 小时多变量天气；
5. 保存输入 CSV/NPY、预测 CSV/JSON/NPZ、单独的结果图和复现报告；
6. 可选在推理阶段启用 FlagGems。

本脚本按 Python 3.10 编写。请保留平台已有的定制 PyTorch，不要使用
pip install torch 或 transformers[torch] 覆盖原环境。
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

MODEL_ID = "Harley-ml/Hweh-6M"
MODEL_REVISION = "710d8df83c41ed4555cd94871f8b64382f514bb8"

CONTEXT_HOURS = 72
FORECAST_HOURS = 12
INPUT_DIM = 22
NUM_WEATHER_CLASSES = 7

TEMP_SCALE = 50.0
HUMIDITY_SCALE = 100.0
WIND_SCALE = 100.0

WEATHER_CLASS_NAMES = [
    "clear",
    "cloudy",
    "fog",
    "drizzle",
    "rain",
    "snow",
    "thunderstorm",
]

INPUT_FEATURE_NAMES = [
    "temperature_2m_norm",
    "relative_humidity_2m_norm",
    "apparent_temperature_norm",
    "precipitation_log_norm",
    "sea_level_pressure_norm",
    "surface_pressure_norm",
    "cloud_cover_total_norm",
    "visibility_norm",
    "wind_speed_10m_norm",
    "wind_direction_10m_sin",
    "wind_direction_10m_cos",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "weather_code_onehot_clear",
    "weather_code_onehot_cloudy",
    "weather_code_onehot_fog",
    "weather_code_onehot_drizzle",
    "weather_code_onehot_rain",
    "weather_code_onehot_snow",
    "weather_code_onehot_thunderstorm",
]

OPEN_METEO_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
    "wind_direction_10m",
]


class ReproductionError(RuntimeError):
    """面向用户的复现错误。"""


def check_python() -> None:
    if sys.version_info[:2] != (3, 10):
        raise ReproductionError(
            f"当前 Python 为 {platform.python_version()}；"
            "本复现方案按 Python 3.10 环境编写。"
        )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_imports() -> dict[str, Any]:
    missing: list[str] = []

    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy")

    try:
        import pandas as pd
    except ImportError:
        pd = None
        missing.append("pandas")

    try:
        import torch
    except ImportError:
        torch = None
        missing.append("torch")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        missing.append("matplotlib")

    try:
        import requests
    except ImportError:
        requests = None
        missing.append("requests")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        snapshot_download = None
        missing.append("huggingface_hub")

    try:
        from transformers import AutoConfig, AutoModel
    except ImportError:
        AutoConfig = None
        AutoModel = None
        missing.append("transformers")

    if missing:
        raise ReproductionError(
            "当前环境缺少以下依赖：\n  - "
            + "\n  - ".join(missing)
            + "\n请先按照 README 中的安装单元格安装依赖。"
        )

    return {
        "np": np,
        "pd": pd,
        "torch": torch,
        "plt": plt,
        "requests": requests,
        "snapshot_download": snapshot_download,
        "AutoConfig": AutoConfig,
        "AutoModel": AutoModel,
    }


def cyc(np: Any, values: Any, period: float) -> tuple[Any, Any]:
    angle = 2.0 * np.pi * (values / period)
    return np.sin(angle), np.cos(angle)


def weather_code_to_bucket(code: Any) -> int:
    """按模型卡给出的 Open-Meteo 天气码分为 7 类。"""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return 1

    if code == 0:
        return 0
    if code in (1, 2, 3):
        return 1
    if code in (45, 48):
        return 2
    if code in (51, 53, 55, 56, 57):
        return 3
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return 4
    if code in (71, 73, 75, 77, 85, 86):
        return 5
    if code in (95, 96, 99):
        return 6
    return 1


def generate_synthetic_history(np: Any, pd: Any, seed: int) -> Any:
    """
    生成具有昼夜周期和连续性的 72 小时仿真天气历史。

    该数据仅用于验证输入构造、模型加载与前向推理，不代表真实城市观测。
    """
    rng = np.random.default_rng(seed)

    end_time = pd.Timestamp("2025-06-15T12:00:00Z")
    times = pd.date_range(
        end=end_time,
        periods=CONTEXT_HOURS,
        freq="h",
        tz="UTC",
    )

    hour = times.hour.to_numpy(dtype=np.float32)
    phase = 2.0 * np.pi * (hour - 6.0) / 24.0
    trend = np.linspace(-0.8, 0.9, CONTEXT_HOURS, dtype=np.float32)

    temperature = (
        18.0
        + 5.5 * np.sin(phase)
        + trend
        + rng.normal(0.0, 0.35, CONTEXT_HOURS)
    )

    humidity = (
        72.0
        - 18.0 * np.sin(phase)
        + rng.normal(0.0, 1.8, CONTEXT_HOURS)
    )
    humidity = np.clip(humidity, 25.0, 100.0)

    apparent_temperature = (
        temperature
        - 0.9
        + 0.018 * (humidity - 60.0)
        + rng.normal(0.0, 0.18, CONTEXT_HOURS)
    )

    rain_center = 55.0
    rain_shape = np.exp(
        -0.5
        * ((np.arange(CONTEXT_HOURS) - rain_center) / 4.5) ** 2
    )
    precipitation = np.clip(
        1.8 * rain_shape + rng.normal(0.0, 0.025, CONTEXT_HOURS),
        0.0,
        None,
    )
    precipitation[precipitation < 0.06] = 0.0

    pressure_msl = (
        1014.0
        + 2.2 * np.sin(
            2.0 * np.pi * np.arange(CONTEXT_HOURS) / 48.0
        )
        - 1.7 * rain_shape
        + rng.normal(0.0, 0.15, CONTEXT_HOURS)
    )

    surface_pressure = (
        pressure_msl
        - 7.5
        + rng.normal(0.0, 0.08, CONTEXT_HOURS)
    )

    cloud_cover = np.clip(
        38.0
        + 50.0 * rain_shape
        + 15.0 * np.cos(phase)
        + rng.normal(0.0, 2.0, CONTEXT_HOURS),
        0.0,
        100.0,
    )

    visibility = np.clip(
        42000.0
        - 26000.0 * rain_shape
        - 120.0 * cloud_cover
        + rng.normal(0.0, 600.0, CONTEXT_HOURS),
        1000.0,
        50000.0,
    )

    wind_speed = np.clip(
        11.0
        + 5.0 * rain_shape
        + 2.0 * np.sin(phase + 0.6)
        + rng.normal(0.0, 0.45, CONTEXT_HOURS),
        0.0,
        None,
    )

    wind_direction = (
        210.0
        + 22.0 * np.sin(
            2.0 * np.pi * np.arange(CONTEXT_HOURS) / 36.0
        )
        + rng.normal(0.0, 2.5, CONTEXT_HOURS)
    ) % 360.0

    weather_code = np.full(CONTEXT_HOURS, 1, dtype=np.int64)
    weather_code[cloud_cover < 25.0] = 0
    weather_code[(precipitation > 0.0) & (precipitation <= 0.35)] = 51
    weather_code[precipitation > 0.35] = 61

    return pd.DataFrame(
        {
            "time": times,
            "temperature_2m": temperature,
            "relative_humidity_2m": humidity,
            "apparent_temperature": apparent_temperature,
            "precipitation": precipitation,
            "weather_code": weather_code,
            "pressure_msl": pressure_msl,
            "surface_pressure": surface_pressure,
            "cloud_cover": cloud_cover,
            "visibility": visibility,
            "wind_speed_10m": wind_speed,
            "wind_direction_10m": wind_direction,
        }
    )


def fetch_open_meteo_history(
    requests: Any,
    pd: Any,
    latitude: float,
    longitude: float,
) -> Any:
    """按模型卡的变量和单位获取最近 72 小时历史天气。"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(OPEN_METEO_HOURLY_VARS),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "past_hours": CONTEXT_HOURS + 2,
        "forecast_hours": 0,
    }

    last_error: Exception | None = None

    for attempt in range(6):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            hourly = payload.get("hourly", {})

            if "time" not in hourly:
                raise ReproductionError(
                    "Open-Meteo 响应中不存在 hourly.time。"
                )

            frame = pd.DataFrame(hourly)
            frame["time"] = pd.to_datetime(
                frame["time"],
                errors="coerce",
                utc=True,
            )
            frame = (
                frame.dropna(subset=["time"])
                .sort_values("time")
                .drop_duplicates(subset=["time"])
                .reset_index(drop=True)
            )

            missing = [
                column
                for column in OPEN_METEO_HOURLY_VARS
                if column not in frame.columns
            ]
            if missing:
                raise ReproductionError(
                    f"Open-Meteo 缺少字段：{missing}"
                )

            for column in OPEN_METEO_HOURLY_VARS:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

            frame["weather_code"] = frame["weather_code"].fillna(1)
            frame["precipitation"] = frame["precipitation"].fillna(0.0)

            for column in [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "pressure_msl",
                "surface_pressure",
                "cloud_cover",
                "visibility",
                "wind_speed_10m",
                "wind_direction_10m",
            ]:
                frame[column] = (
                    frame[column]
                    .interpolate(limit_direction="both")
                    .ffill()
                    .bfill()
                )

            now_utc = pd.Timestamp.now(tz="UTC")
            frame = frame[frame["time"] <= now_utc].copy()

            if len(frame) < CONTEXT_HOURS:
                raise ReproductionError(
                    f"真实历史数据只有 {len(frame)} 行，"
                    f"模型需要 {CONTEXT_HOURS} 行。"
                )

            return frame.tail(CONTEXT_HOURS).reset_index(drop=True)

        except Exception as exc:
            last_error = exc
            if attempt == 5:
                break
            time.sleep(min(2**attempt, 30))

    raise ReproductionError(
        "Open-Meteo 历史数据获取失败。"
    ) from last_error


def build_model_sequence(np: Any, pd: Any, frame: Any) -> Any:
    """
    按模型卡的公式构造 72×22 输入特征。
    """
    if len(frame) != CONTEXT_HOURS:
        raise ReproductionError(
            f"历史序列长度必须为 {CONTEXT_HOURS}，"
            f"实际为 {len(frame)}。"
        )

    hour = frame["time"].dt.hour.to_numpy(dtype=np.float32)
    day_of_year = frame["time"].dt.dayofyear.to_numpy(dtype=np.float32)

    hour_sin, hour_cos = cyc(np, hour, 24.0)
    doy_sin, doy_cos = cyc(np, day_of_year, 365.25)

    temperature = np.nan_to_num(
        frame["temperature_2m"].to_numpy(dtype=np.float32),
        nan=0.0,
    )
    humidity = np.clip(
        np.nan_to_num(
            frame["relative_humidity_2m"].to_numpy(dtype=np.float32),
            nan=0.0,
        ),
        0.0,
        100.0,
    )
    apparent = np.nan_to_num(
        frame["apparent_temperature"].to_numpy(dtype=np.float32),
        nan=0.0,
    )
    precipitation = np.clip(
        np.nan_to_num(
            frame["precipitation"].to_numpy(dtype=np.float32),
            nan=0.0,
        ),
        0.0,
        None,
    )
    pressure_msl = np.nan_to_num(
        frame["pressure_msl"].to_numpy(dtype=np.float32),
        nan=0.0,
    )
    surface_pressure = np.nan_to_num(
        frame["surface_pressure"].to_numpy(dtype=np.float32),
        nan=0.0,
    )
    cloud_cover = np.clip(
        np.nan_to_num(
            frame["cloud_cover"].to_numpy(dtype=np.float32),
            nan=0.0,
        ),
        0.0,
        100.0,
    )
    visibility = np.clip(
        np.nan_to_num(
            frame["visibility"].to_numpy(dtype=np.float32),
            nan=0.0,
        ),
        0.0,
        None,
    )
    wind_speed = np.clip(
        np.nan_to_num(
            frame["wind_speed_10m"].to_numpy(dtype=np.float32),
            nan=0.0,
        ),
        0.0,
        None,
    )
    wind_direction = np.nan_to_num(
        frame["wind_direction_10m"].to_numpy(dtype=np.float32),
        nan=0.0,
    )

    wind_sin, wind_cos = cyc(np, wind_direction, 360.0)

    buckets = np.asarray(
        [
            weather_code_to_bucket(code)
            for code in frame["weather_code"].tolist()
        ],
        dtype=np.int64,
    )
    one_hot = np.zeros(
        (CONTEXT_HOURS, NUM_WEATHER_CLASSES),
        dtype=np.float32,
    )
    one_hot[np.arange(CONTEXT_HOURS), buckets] = 1.0

    continuous = np.column_stack(
        [
            temperature / TEMP_SCALE,
            humidity / HUMIDITY_SCALE,
            apparent / TEMP_SCALE,
            np.log1p(precipitation) / 3.0,
            pressure_msl / 1100.0,
            surface_pressure / 1100.0,
            cloud_cover / 100.0,
            visibility / 50000.0,
            wind_speed / WIND_SCALE,
            wind_sin,
            wind_cos,
            hour_sin,
            hour_cos,
            doy_sin,
            doy_cos,
        ]
    ).astype(np.float32)

    sequence = np.concatenate(
        [continuous, one_hot],
        axis=1,
    ).astype(np.float32)

    if sequence.shape != (CONTEXT_HOURS, INPUT_DIM):
        raise ReproductionError(
            "输入特征形状错误："
            f"{sequence.shape} != {(CONTEXT_HOURS, INPUT_DIM)}"
        )

    if not np.isfinite(sequence).all():
        raise ReproductionError("输入特征中存在 NaN 或 Inf。")

    return sequence


def ensure_model_files(
    snapshot_download: Any,
    model_dir: Path,
    offline: bool,
) -> Path:
    required = [
        "config.json",
        "configuration.py",
        "modeling.py",
        "model.safetensors",
    ]

    if all((model_dir / name).exists() for name in required):
        print("模型文件已存在，跳过下载。")
        return model_dir

    if offline:
        raise ReproductionError(
            "指定了 --offline，但 model_files 中缺少完整模型文件。"
        )

    print("正在从 Hugging Face 下载固定版本模型……")
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(model_dir),
        allow_patterns=[
            "README.md",
            "__init__.py",
            "config.json",
            "configuration.py",
            "modeling.py",
            "model.safetensors",
        ],
    )

    missing = [
        name
        for name in required
        if not (model_dir / name).exists()
    ]
    if missing:
        raise ReproductionError(
            f"模型下载后仍缺少文件：{missing}"
        )

    return model_dir


def patch_model_configuration(model_dir: Path) -> dict[str, Any]:
    """
    Repair the undefined ``distill_teacher_head_dim`` local variable.

    The pinned upstream configuration reads this name inside ``__init__``
    without necessarily declaring it as a parameter or assigning it first.
    This implementation parses the downloaded Python source and inserts a
    local fallback only when the name is genuinely loaded but undefined.
    """
    config_path = model_dir / "configuration.py"
    if not config_path.is_file():
        raise ReproductionError(
            f"无法修复模型配置，文件不存在：{config_path}"
        )

    original = config_path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(original, filename=str(config_path))
    except SyntaxError as exc:
        raise ReproductionError(
            f"configuration.py 无法解析：{exc}"
        ) from exc

    init_function = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "__init__":
                init_function = node
                break

    if init_function is None:
        raise ReproductionError(
            "configuration.py 中没有找到 __init__，无法应用兼容修复。"
        )

    argument_names = {
        argument.arg
        for argument in (
            list(init_function.args.posonlyargs)
            + list(init_function.args.args)
            + list(init_function.args.kwonlyargs)
        )
    }
    if init_function.args.vararg is not None:
        argument_names.add(init_function.args.vararg.arg)
    if init_function.args.kwarg is not None:
        argument_names.add(init_function.args.kwarg.arg)

    target_name = "distill_teacher_head_dim"

    load_nodes = [
        node
        for node in ast.walk(init_function)
        if isinstance(node, ast.Name)
        and node.id == target_name
        and isinstance(node.ctx, ast.Load)
    ]
    store_nodes = [
        node
        for node in ast.walk(init_function)
        if isinstance(node, ast.Name)
        and node.id == target_name
        and isinstance(node.ctx, (ast.Store, ast.Param))
    ]

    if target_name in argument_names or store_nodes:
        print(
            "configuration.py 中 distill_teacher_head_dim "
            "已经有参数或局部定义，无需修复。"
        )
        return {
            "path": str(config_path),
            "status": "already_defined",
            "replacement_count": 0,
        }

    if not load_nodes:
        print(
            "configuration.py 未读取未定义的 "
            "distill_teacher_head_dim，无需修复。"
        )
        return {
            "path": str(config_path),
            "status": "not_required",
            "replacement_count": 0,
        }

    if init_function.args.kwarg is None:
        raise ReproductionError(
            "configuration.py 的 __init__ 没有 **kwargs，"
            "无法安全读取 distill_teacher_head_dim。"
        )

    hidden_dim_available = (
        "hidden_dim" in argument_names
        or any(
            isinstance(node, ast.Name)
            and node.id == "hidden_dim"
            and isinstance(node.ctx, ast.Store)
            for node in ast.walk(init_function)
        )
    )
    if not hidden_dim_available:
        raise ReproductionError(
            "configuration.py 中没有找到 hidden_dim，"
            "无法建立 distill_teacher_head_dim 的默认值。"
        )

    if not init_function.body:
        raise ReproductionError(
            "configuration.py 的 __init__ 函数体为空。"
        )

    first_statement = init_function.body[0]
    insert_line_index = first_statement.lineno - 1
    source_lines = original.splitlines(keepends=True)

    first_line = source_lines[insert_line_index]
    indent = first_line[: len(first_line) - len(first_line.lstrip())]
    kwargs_name = init_function.args.kwarg.arg

    insertion = [
        (
            f"{indent}# Compatibility fix for "
            "Harley-ml/Hweh-6M configuration.\n"
        ),
        (
            f'{indent}{target_name} = {kwargs_name}.pop('
            f'"{target_name}", hidden_dim)\n'
        ),
    ]
    source_lines[insert_line_index:insert_line_index] = insertion
    patched = "".join(source_lines)

    try:
        patched_tree = ast.parse(patched, filename=str(config_path))
        compile(patched_tree, str(config_path), "exec")
    except (SyntaxError, ValueError) as exc:
        raise ReproductionError(
            f"自动修复后的 configuration.py 无法编译：{exc}"
        ) from exc

    config_path.write_text(
        patched,
        encoding="utf-8",
        newline="\n",
    )
    print(
        "已自动修复 configuration.py：在 __init__ 开头定义 "
        "distill_teacher_head_dim。"
    )

    return {
        "path": str(config_path),
        "status": "patched_by_ast",
        "replacement_count": 1,
        "inserted_before_line": first_statement.lineno,
    }


def reset_transformers_dynamic_cache(base_dir: Path) -> Path:
    """
    Use a project-local Transformers dynamic-module cache and clear stale code.
    """
    cache_dir = base_dir / "hf_modules_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("Transformers 动态模块缓存已重置：", cache_dir)
    return cache_dir


def load_model(
    AutoConfig: Any,
    AutoModel: Any,
    model_dir: Path,
) -> tuple[Any, Any]:
    """
    仓库包含自定义 configuration.py 和 modeling.py，因此必须
    trust_remote_code=True。模型固定到已审查的提交并从本地目录加载。
    """
    config = AutoConfig.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(model_dir),
        config=config,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return model, config


def extract_logits(output: Any) -> Any:
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if hasattr(output, "logits"):
        return output.logits
    return output


def tensor_vector(
    np: Any,
    tensor: Any,
    name: str,
    expected: int = FORECAST_HOURS,
) -> Any:
    array = tensor.detach().float().cpu().numpy()
    array = np.squeeze(array)

    if array.ndim != 1 or array.shape[0] != expected:
        array = array.reshape(-1)

    if array.shape != (expected,):
        raise ReproductionError(
            f"{name} 输出形状异常：{array.shape}"
        )

    return array.astype(np.float32)


def tensor_weather_logits(np: Any, tensor: Any) -> Any:
    array = tensor.detach().float().cpu().numpy()
    array = np.squeeze(array)

    expected = (FORECAST_HOURS, NUM_WEATHER_CLASSES)
    if array.shape != expected:
        try:
            array = array.reshape(expected)
        except ValueError as exc:
            raise ReproductionError(
                f"weather_logits 输出形状异常：{array.shape}"
            ) from exc

    return array.astype(np.float32)


def softmax_numpy(np: Any, logits: Any) -> Any:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def sigmoid_numpy(np: Any, values: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-values))


def prepare_flag_gems(enabled: bool) -> tuple[Any, dict[str, Any]]:
    info = {
        "requested": bool(enabled),
        "enabled": False,
        "mode": None,
    }

    if not enabled:
        return contextlib.nullcontext(), info

    try:
        import flag_gems
    except ImportError as exc:
        raise ReproductionError(
            "指定了 --flag-gems，但当前环境没有 flag_gems。"
        ) from exc

    if hasattr(flag_gems, "use_gems"):
        info.update(
            {
                "enabled": True,
                "mode": "use_gems context manager",
            }
        )
        return flag_gems.use_gems(), info

    flag_gems.enable(
        unused=[
            "batch_norm",
            "batch_norm_backward",
        ]
    )
    info.update(
        {
            "enabled": True,
            "mode": "global enable fallback",
        }
    )
    return contextlib.nullcontext(), info


def run_model(
    np: Any,
    torch: Any,
    model: Any,
    sequence: Any,
    location_index: int,
    device: str,
    use_flag_gems: bool,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    input_tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
    location_tensor = torch.tensor(
        [location_index],
        dtype=torch.long,
        device=device,
    )

    gems_context, gems_info = prepare_flag_gems(use_flag_gems)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        with gems_context:
            output = model(
                X=input_tensor,
                location_id=location_tensor,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    logits = extract_logits(output)

    if not isinstance(logits, (tuple, list)) or len(logits) != 12:
        raise ReproductionError(
            "模型应返回 12 个任务头，实际返回："
            f"{type(logits).__name__}"
        )

    (
        temperature,
        humidity,
        apparent_temperature,
        precipitation,
        pressure_msl,
        surface_pressure,
        cloud_cover,
        wind_speed,
        wind_direction_sin,
        wind_direction_cos,
        rain_logit,
        weather_logits,
    ) = logits

    outputs = {
        "temperature_2m_c": tensor_vector(
            np, temperature, "temperature"
        ),
        "relative_humidity_2m_pct": tensor_vector(
            np, humidity, "humidity"
        ),
        "apparent_temperature_c": tensor_vector(
            np, apparent_temperature, "apparent_temperature"
        ),
        "precipitation_mm": tensor_vector(
            np, precipitation, "precipitation"
        ),
        "pressure_msl_hpa": tensor_vector(
            np, pressure_msl, "pressure_msl"
        ),
        "surface_pressure_hpa": tensor_vector(
            np, surface_pressure, "surface_pressure"
        ),
        "cloud_cover_pct": tensor_vector(
            np, cloud_cover, "cloud_cover"
        ),
        "wind_speed_10m_kmh": tensor_vector(
            np, wind_speed, "wind_speed"
        ),
        "wind_direction_sin": tensor_vector(
            np, wind_direction_sin, "wind_direction_sin"
        ),
        "wind_direction_cos": tensor_vector(
            np, wind_direction_cos, "wind_direction_cos"
        ),
        "rain_logit": tensor_vector(
            np, rain_logit, "rain_logit"
        ),
        "weather_logits": tensor_weather_logits(
            np, weather_logits
        ),
    }

    outputs["relative_humidity_2m_pct"] = np.clip(
        outputs["relative_humidity_2m_pct"],
        0.0,
        100.0,
    )
    outputs["precipitation_mm"] = np.clip(
        outputs["precipitation_mm"],
        0.0,
        None,
    )
    outputs["cloud_cover_pct"] = np.clip(
        outputs["cloud_cover_pct"],
        0.0,
        100.0,
    )
    outputs["wind_speed_10m_kmh"] = np.clip(
        outputs["wind_speed_10m_kmh"],
        0.0,
        None,
    )
    outputs["rain_probability"] = np.clip(
        sigmoid_numpy(np, outputs["rain_logit"]),
        0.0,
        1.0,
    )
    outputs["weather_probabilities"] = softmax_numpy(
        np,
        outputs["weather_logits"],
    )
    outputs["weather_class"] = np.argmax(
        outputs["weather_probabilities"],
        axis=-1,
    ).astype(np.int64)
    outputs["wind_direction_10m_deg"] = (
        np.degrees(
            np.arctan2(
                outputs["wind_direction_sin"],
                outputs["wind_direction_cos"],
            )
        )
        + 360.0
    ) % 360.0

    return outputs, elapsed, gems_info


def save_line_plot(
    plt: Any,
    x_values: Any,
    y_values: Any,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(8.0, 4.6))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(x_values, y_values, marker="o")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_plots(
    plt: Any,
    history: Any,
    forecast_frame: Any,
    output_dir: Path,
) -> list[str]:
    files: list[str] = []

    plot_specs = [
        (
            history["time"],
            history["temperature_2m"],
            "Synthetic/Observed 72-hour Temperature History",
            "Time",
            "Temperature (°C)",
            "input_temperature_history.png",
        ),
        (
            forecast_frame["lead_hours"],
            forecast_frame["temperature_2m_c"],
            "Hweh-6M 12-hour Temperature Forecast",
            "Lead time (hours)",
            "Temperature (°C)",
            "forecast_temperature.png",
        ),
        (
            forecast_frame["lead_hours"],
            forecast_frame["relative_humidity_2m_pct"],
            "Hweh-6M 12-hour Humidity Forecast",
            "Lead time (hours)",
            "Relative humidity (%)",
            "forecast_humidity.png",
        ),
        (
            forecast_frame["lead_hours"],
            forecast_frame["precipitation_mm"],
            "Hweh-6M 12-hour Precipitation Forecast",
            "Lead time (hours)",
            "Precipitation (mm)",
            "forecast_precipitation.png",
        ),
        (
            forecast_frame["lead_hours"],
            forecast_frame["pressure_msl_hpa"],
            "Hweh-6M 12-hour Sea-level Pressure Forecast",
            "Lead time (hours)",
            "Pressure (hPa)",
            "forecast_pressure.png",
        ),
        (
            forecast_frame["lead_hours"],
            forecast_frame["rain_probability"],
            "Hweh-6M 12-hour Rain Probability",
            "Lead time (hours)",
            "Probability",
            "forecast_rain_probability.png",
        ),
        (
            forecast_frame["lead_hours"],
            forecast_frame["wind_speed_10m_kmh"],
            "Hweh-6M 12-hour Wind-speed Forecast",
            "Lead time (hours)",
            "Wind speed (km/h)",
            "forecast_wind_speed.png",
        ),
    ]

    for x_values, y_values, title, x_label, y_label, filename in plot_specs:
        path = output_dir / filename
        save_line_plot(
            plt,
            x_values,
            y_values,
            title,
            x_label,
            y_label,
            path,
        )
        files.append(str(path))

    return files


def build_forecast_frame(
    np: Any,
    pd: Any,
    outputs: dict[str, Any],
    history: Any,
    weather_names: list[str],
) -> Any:
    lead_hours = np.arange(1, FORECAST_HOURS + 1, dtype=np.int64)
    last_time = history["time"].iloc[-1]
    target_times = [
        last_time + pd.Timedelta(hours=int(lead))
        for lead in lead_hours
    ]

    weather_class = outputs["weather_class"]
    weather_name = [
        weather_names[int(index)]
        if 0 <= int(index) < len(weather_names)
        else f"class_{int(index)}"
        for index in weather_class
    ]

    data = {
        "lead_hours": lead_hours,
        "target_utc": [time_value.isoformat() for time_value in target_times],
        "temperature_2m_c": outputs["temperature_2m_c"],
        "relative_humidity_2m_pct": outputs[
            "relative_humidity_2m_pct"
        ],
        "apparent_temperature_c": outputs[
            "apparent_temperature_c"
        ],
        "precipitation_mm": outputs["precipitation_mm"],
        "pressure_msl_hpa": outputs["pressure_msl_hpa"],
        "surface_pressure_hpa": outputs["surface_pressure_hpa"],
        "cloud_cover_pct": outputs["cloud_cover_pct"],
        "wind_speed_10m_kmh": outputs["wind_speed_10m_kmh"],
        "wind_direction_10m_deg": outputs[
            "wind_direction_10m_deg"
        ],
        "rain_probability": outputs["rain_probability"],
        "weather_class": weather_class,
        "weather_class_name": weather_name,
    }

    for index, name in enumerate(weather_names):
        data[f"weather_probability_{name}"] = outputs[
            "weather_probabilities"
        ][:, index]

    return pd.DataFrame(data)


def all_finite(np: Any, outputs: dict[str, Any]) -> bool:
    for value in outputs.values():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harley-ml/Hweh-6M 一键推理复现"
    )
    parser.add_argument(
        "--data-mode",
        choices=["synthetic", "open-meteo"],
        default="synthetic",
        help="默认 synthetic；真实历史数据可选 open-meteo。",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=47.6062,
        help="Open-Meteo 模式纬度，默认 Seattle。",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=-122.3321,
        help="Open-Meteo 模式经度，默认 Seattle。",
    )
    parser.add_argument(
        "--location-index",
        type=int,
        default=0,
        help="模型位置嵌入索引，必须为 0–81；默认 0。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="仿真数据随机种子。",
    )
    parser.add_argument(
        "--flag-gems",
        action="store_true",
        help="仅在模型前向推理阶段启用 FlagGems。",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="强制使用 CPU。",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="只使用 model_files 中已有文件，不联网下载。",
    )
    args = parser.parse_args()

    check_python()

    if not 0 <= args.location_index < 82:
        raise ReproductionError(
            "--location-index 必须位于 0–81。"
        )

    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model_files"
    output_dir = base_dir / "hweh_outputs"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # These variables must be set before importing Transformers/HF Hub.
    os.environ["HF_HOME"] = str(base_dir / "hf_cache")
    os.environ["HF_MODULES_CACHE"] = str(base_dir / "hf_modules_cache")

    modules = check_imports()
    np = modules["np"]
    pd = modules["pd"]
    torch = modules["torch"]
    plt = modules["plt"]

    print("=" * 72)
    print("Hweh-6M 推理复现")
    print("=" * 72)
    print("Python：", platform.python_version())
    print("Python 路径：", sys.executable)
    print("PyTorch：", torch.__version__)
    print("设备可用：", torch.cuda.is_available())
    print("数据模式：", args.data_mode)
    print("Hugging Face 模型：", MODEL_ID)
    print("模型提交：", MODEL_REVISION)

    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print("运行设备：", device)
    if device.startswith("cuda"):
        print("设备名称：", torch.cuda.get_device_name(0))

    ensure_model_files(
        modules["snapshot_download"],
        model_dir,
        args.offline,
    )
    configuration_patch = patch_model_configuration(model_dir)
    reset_transformers_dynamic_cache(base_dir)

    if args.data_mode == "synthetic":
        history = generate_synthetic_history(
            np,
            pd,
            args.seed,
        )
        data_source = "programmatically generated synthetic weather"
    else:
        history = fetch_open_meteo_history(
            modules["requests"],
            pd,
            args.latitude,
            args.longitude,
        )
        data_source = "Open-Meteo recent 72-hour history"

    sequence = build_model_sequence(np, pd, history)

    history_csv = output_dir / "input_history.csv"
    normalized_csv = output_dir / "input_features_normalized.csv"
    normalized_npy = output_dir / "input_features_normalized.npy"

    history.to_csv(history_csv, index=False)
    pd.DataFrame(
        sequence,
        columns=INPUT_FEATURE_NAMES,
    ).to_csv(normalized_csv, index=False)
    np.save(normalized_npy, sequence)

    print("输入原始数据：", history_csv)
    print("输入特征形状：", sequence.shape)
    print("输入特征有限：", bool(np.isfinite(sequence).all()))

    print("\n正在加载模型……")
    load_start = time.perf_counter()
    model, config = load_model(
        modules["AutoConfig"],
        modules["AutoModel"],
        model_dir,
    )
    model = model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_start

    parameter_count = sum(
        int(parameter.numel())
        for parameter in model.parameters()
    )

    print(f"模型加载完成：{load_seconds:.3f} 秒")
    print("参数量：", parameter_count)

    outputs, inference_seconds, gems_info = run_model(
        np,
        torch,
        model,
        sequence,
        args.location_index,
        device,
        args.flag_gems,
    )

    configured_names = getattr(
        config,
        "weather_class_names",
        None,
    )
    weather_names = (
        list(configured_names)
        if configured_names
        and len(configured_names) == NUM_WEATHER_CLASSES
        else WEATHER_CLASS_NAMES
    )

    forecast_frame = build_forecast_frame(
        np,
        pd,
        outputs,
        history,
        weather_names,
    )

    forecast_csv = output_dir / "forecast_12h.csv"
    forecast_json = output_dir / "forecast_12h.json"
    raw_npz = output_dir / "raw_model_outputs.npz"

    forecast_frame.to_csv(forecast_csv, index=False)
    forecast_json.write_text(
        forecast_frame.to_json(
            orient="records",
            force_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    np.savez(
        raw_npz,
        **{
            name: value
            for name, value in outputs.items()
            if isinstance(value, np.ndarray)
        },
    )

    plot_files = save_plots(
        plt,
        history,
        forecast_frame,
        output_dir,
    )

    finite_outputs = all_finite(np, outputs)

    report = {
        "success": bool(
            sequence.shape == (CONTEXT_HOURS, INPUT_DIM)
            and len(forecast_frame) == FORECAST_HOURS
            and finite_outputs
        ),
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "parameter_count": parameter_count,
            "encoder_type": getattr(config, "encoder_type", None),
            "input_dim": int(getattr(config, "input_dim", INPUT_DIM)),
            "seq_len": int(
                getattr(config, "seq_len", CONTEXT_HOURS)
            ),
            "num_predict": int(
                getattr(config, "num_predict", FORECAST_HOURS)
            ),
            "num_locations": int(
                getattr(config, "num_locations", 82)
            ),
            "num_weather_classes": int(
                getattr(
                    config,
                    "num_weather_classes",
                    NUM_WEATHER_CLASSES,
                )
            ),
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": device,
            "device_name": (
                torch.cuda.get_device_name(0)
                if device.startswith("cuda")
                else None
            ),
            "packages": {
                name: package_version(name)
                for name in [
                    "torch",
                    "transformers",
                    "huggingface-hub",
                    "safetensors",
                    "numpy",
                    "pandas",
                    "matplotlib",
                    "requests",
                    "flag-gems",
                ]
            },
        },
        "input": {
            "data_mode": args.data_mode,
            "data_source": data_source,
            "latitude": (
                args.latitude
                if args.data_mode == "open-meteo"
                else None
            ),
            "longitude": (
                args.longitude
                if args.data_mode == "open-meteo"
                else None
            ),
            "location_index": args.location_index,
            "raw_rows": int(len(history)),
            "normalized_shape": list(sequence.shape),
            "normalized_dtype": str(sequence.dtype),
            "finite": bool(np.isfinite(sequence).all()),
        },
        "output": {
            "forecast_rows": int(len(forecast_frame)),
            "finite": finite_outputs,
            "temperature_min_c": float(
                forecast_frame["temperature_2m_c"].min()
            ),
            "temperature_max_c": float(
                forecast_frame["temperature_2m_c"].max()
            ),
            "rain_probability_min": float(
                forecast_frame["rain_probability"].min()
            ),
            "rain_probability_max": float(
                forecast_frame["rain_probability"].max()
            ),
        },
        "timing": {
            "model_load_seconds": round(load_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
        },
        "flag_gems": gems_info,
        "compatibility": {
            "configuration_patch": configuration_patch,
            "hf_modules_cache": str(base_dir / "hf_modules_cache"),
        },
        "files": {
            "model_dir": str(model_dir),
            "input_history_csv": str(history_csv),
            "input_normalized_csv": str(normalized_csv),
            "input_normalized_npy": str(normalized_npy),
            "forecast_csv": str(forecast_csv),
            "forecast_json": str(forecast_json),
            "raw_outputs_npz": str(raw_npz),
            "plots": plot_files,
        },
        "warning": (
            "synthetic 模式只验证工程推理流程。真实天气预测必须使用"
            "正确的最近 72 小时历史数据和合适的位置嵌入索引；"
            "本模型不适用于安全关键业务。"
        ),
    }

    report_path = output_dir / "reproduction_report.json"
    report["files"]["report_json"] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("复现完成")
    print("=" * 72)
    print("预测行数：", len(forecast_frame))
    print("输出全部有限：", finite_outputs)
    print("推理耗时：", f"{inference_seconds:.6f} 秒")
    print("预测 CSV：", forecast_csv)
    print("预测 JSON：", forecast_json)
    print("复现报告：", report_path)
    print("\n前 3 小时预测：")
    print(forecast_frame.head(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中止运行。", file=sys.stderr)
        raise SystemExit(130)
    except ReproductionError as exc:
        print(f"\n复现未完成：{exc}", file=sys.stderr)
        raise SystemExit(1)
