#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import time
import argparse
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from zoneinfo import ZoneInfo


# ============================================================
# 0. 兼容性补丁
# ============================================================

# 兼容 transformers 加载自定义模型时可能需要的 register_for_auto_class
if not hasattr(torch.nn.Module, "register_for_auto_class"):
    torch.nn.Module.register_for_auto_class = classmethod(
        lambda cls, auto_class=None: None
    )

# 兼容模型初始化时可能调用的 post_init
if not hasattr(torch.nn.Module, "post_init"):
    torch.nn.Module.post_init = lambda self: None


# ============================================================
# 1. 启用 flag_gems，适配曙光 DCU 环境
# ============================================================

try:
    import flag_gems

    flag_gems.enable(
        unused=["batch_norm", "batch_norm_backward"],
        record=False,
        once=True
    )
    print("[OK] flag_gems enabled.", flush=True)
except Exception as e:
    print("[WARN] flag_gems not enabled:", repr(e), flush=True)


# ============================================================
# 2. 基本配置
# ============================================================

MODEL_ID = "Harley-ml/Hweh-446k"

# 模型仓库本地备份目录
# 第一次运行脚本时，会自动把 Hugging Face 模型文件下载到这里
LOCAL_MODEL_DIR = "/root/private_data/hweh_446k_test/hf_model_backup"

CONTEXT_HOURS = 72
FORECAST_HOURS = 12
DEVICE = None  # None 表示自动选择 cuda / cpu

API_BASE_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 6

HOURLY_VARS = [
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

WEATHER_CODE_BUCKETS = 7

TEMP_SCALE = 50.0
HUMIDITY_SCALE = 100.0
WIND_SCALE = 100.0


# ============================================================
# 3. 城市信息
# ============================================================

CITY_SPECS: dict[str, dict[str, Any]] = {
    "Seattle": {
        "location_id": "1",
        "latitude": 47.6062,
        "longitude": -122.3321,
        "elevation": 56,
    },
    "New York": {
        "location_id": "9",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "elevation": 10,
    },
    "Singapore": {
        "location_id": "56",
        "latitude": 1.3521,
        "longitude": 103.8198,
        "elevation": 15,
    },
}

CITY_TIMEZONES: dict[str, str] = {
    "Seattle": "America/Los_Angeles",
    "New York": "America/New_York",
    "Singapore": "Asia/Singapore",
}


# ============================================================
# 4. 数据处理函数
# ============================================================

def weather_code_to_bucket(code) -> int:
    if code is None:
        return 1

    try:
        if pd.isna(code):
            return 1
    except Exception:
        pass

    code = int(code)

    if code == 0:
        return 0  # clear
    if code in (1, 2, 3):
        return 1  # cloudy
    if code in (45, 48):
        return 2  # fog
    if code in (51, 53, 55, 56, 57):
        return 3  # drizzle
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return 4  # rain
    if code in (71, 73, 75, 77, 85, 86):
        return 5  # snow
    if code in (95, 96, 99):
        return 6  # thunderstorm

    return 1


def cyc(x: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * (x / period)
    return np.sin(angle), np.cos(angle)


def request_with_backoff(url: str, params: dict[str, Any]) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    last_exc = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT_S)

            if resp.status_code == 429:
                sleep_s = min(60.0, 2 ** attempt)
                print(f"[WARN] Rate limited. Sleep {sleep_s:.1f}s and retry.", flush=True)
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            last_exc = e
            sleep_s = min(60.0, 2 ** attempt)
            print(f"[WARN] Request failed: {e}. Sleep {sleep_s:.1f}s and retry.", flush=True)
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries.") from last_exc


def fetch_recent_history(city: str, context_hours: int) -> pd.DataFrame:
    if city not in CITY_SPECS:
        raise ValueError(f"Unknown city: {city}. Available cities: {list(CITY_SPECS.keys())}")

    spec = CITY_SPECS[city]

    params = {
        "latitude": spec["latitude"],
        "longitude": spec["longitude"],
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "past_hours": int(context_hours) + 2,
        "forecast_hours": 0,
    }

    data = request_with_backoff(API_BASE_URL, params=params)

    hourly = data.get("hourly", {})
    if "time" not in hourly:
        raise ValueError(f"No hourly data returned for {city}: {data}")

    df = pd.DataFrame(hourly)

    if df.empty:
        raise ValueError(f"Empty hourly response for {city}.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = (
        df.dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )

    missing = [c for c in HOURLY_VARS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing hourly columns in API response: {missing}")

    for c in HOURLY_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["weather_code"] = df["weather_code"].fillna(1)
    df["precipitation"] = df["precipitation"].fillna(0.0)

    for c in [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "visibility",
        "wind_speed_10m",
        "wind_direction_10m",
    ]:
        df[c] = df[c].interpolate(limit_direction="both").ffill().bfill()

    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df["time"] <= now_utc].copy()

    if len(df) < context_hours:
        raise ValueError(f"Not enough observed rows: got {len(df)}, need {context_hours}")

    return df.tail(context_hours).reset_index(drop=True)


def build_single_sequence(df: pd.DataFrame) -> np.ndarray:
    hour = df["time"].dt.hour.to_numpy()
    doy = df["time"].dt.dayofyear.to_numpy()

    hour_sin, hour_cos = cyc(hour.astype(float), 24.0)
    doy_sin, doy_cos = cyc(doy.astype(float), 365.25)

    temp = np.nan_to_num(df["temperature_2m"].astype(float).to_numpy(), nan=0.0)
    humidity = np.nan_to_num(df["relative_humidity_2m"].astype(float).to_numpy(), nan=0.0)
    apparent = np.nan_to_num(df["apparent_temperature"].astype(float).to_numpy(), nan=0.0)
    precip = np.nan_to_num(df["precipitation"].astype(float).to_numpy(), nan=0.0)
    pressure = np.nan_to_num(df["pressure_msl"].astype(float).to_numpy(), nan=0.0)
    surface_pressure = np.nan_to_num(df["surface_pressure"].astype(float).to_numpy(), nan=0.0)
    cloud_cover = np.nan_to_num(df["cloud_cover"].astype(float).to_numpy(), nan=0.0)
    visibility = np.nan_to_num(df["visibility"].astype(float).to_numpy(), nan=0.0)
    wind = np.nan_to_num(df["wind_speed_10m"].astype(float).to_numpy(), nan=0.0)
    wind_dir = np.nan_to_num(df["wind_direction_10m"].astype(float).to_numpy(), nan=0.0)

    humidity = np.clip(humidity, 0.0, 100.0)
    cloud_cover = np.clip(cloud_cover, 0.0, 100.0)
    precip = np.clip(precip, 0.0, None)
    wind = np.clip(wind, 0.0, None)
    visibility = np.clip(visibility, 0.0, None)

    wind_dir_sin, wind_dir_cos = cyc(wind_dir, 360.0)

    weather_bucket = (
        df["weather_code"]
        .fillna(1)
        .apply(weather_code_to_bucket)
        .to_numpy(dtype=np.int64)
    )

    rows = []

    for i in range(len(df)):
        weather_onehot = np.zeros(WEATHER_CODE_BUCKETS, dtype=np.float32)
        weather_onehot[weather_bucket[i]] = 1.0

        row = np.concatenate(
            [
                np.array(
                    [
                        temp[i] / TEMP_SCALE,
                        humidity[i] / HUMIDITY_SCALE,
                        apparent[i] / TEMP_SCALE,
                        np.log1p(max(precip[i], 0.0)) / 3.0,
                        pressure[i] / 1100.0,
                        surface_pressure[i] / 1100.0,
                        cloud_cover[i] / 100.0,
                        visibility[i] / 50000.0,
                        wind[i] / WIND_SCALE,
                        wind_dir_sin[i],
                        wind_dir_cos[i],
                        hour_sin[i],
                        hour_cos[i],
                        doy_sin[i],
                        doy_cos[i],
                    ],
                    dtype=np.float32,
                ),
                weather_onehot,
            ]
        )

        rows.append(row)

    seq = np.asarray(rows, dtype=np.float32)

    if not np.isfinite(seq).all():
        bad = np.argwhere(~np.isfinite(seq))
        raise ValueError(f"Non-finite values remain in sequence: {bad[:10].tolist()}")

    return seq


def to_iso(ts: pd.Timestamp, tz_name: str | None = None) -> str:
    if tz_name:
        try:
            return ts.tz_convert(ZoneInfo(tz_name)).isoformat()
        except Exception:
            pass
    return ts.isoformat()


# ============================================================
# 5. 手动加载模型：下载模型文件、加载模型类、加载权重
# ============================================================

def load_model_manual(model_id: str):
    print("\n[Step 1] Downloading / locating model snapshot...", flush=True)
    t0 = time.time()

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    snapshot_dir = snapshot_download(
        repo_id=model_id,
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json",
            "configuration.py",
            "modeling.py",
            "*.bin",
            "*.safetensors",
            "*.json",
        ],
    )

    print("Local model directory:", LOCAL_MODEL_DIR, flush=True)
    print("Snapshot directory:", snapshot_dir, flush=True)
    print("Snapshot ready. Time: {:.3f} s".format(time.time() - t0), flush=True)

    print("\n[Step 2] Loading config and model class...", flush=True)

    config = AutoConfig.from_pretrained(
        snapshot_dir,
        trust_remote_code=True,
    )

    auto_map = getattr(config, "auto_map", {})
    print("auto_map:", auto_map, flush=True)

    if isinstance(auto_map, dict) and "AutoModel" in auto_map:
        class_ref = auto_map["AutoModel"]
    else:
        class_ref = "modeling.WeatherForcastModel"

    print("Model class reference:", class_ref, flush=True)

    ModelClass = get_class_from_dynamic_module(
        class_ref,
        snapshot_dir,
        trust_remote_code=True,
    )

    print("Model class:", ModelClass, flush=True)

    print("\n[Step 3] Instantiating model...", flush=True)

    model = ModelClass(config)

    print("[OK] Model instantiated.", flush=True)

    print("\n[Step 4] Loading model weights...", flush=True)

    weight_files = []

    for root, dirs, files in os.walk(snapshot_dir):
        for f in files:
            if f.endswith(".safetensors") or f.endswith(".bin"):
                weight_files.append(os.path.join(root, f))

    if not weight_files:
        raise FileNotFoundError("No .bin or .safetensors weight file found.")

    print("Found weight files:", flush=True)
    for p in weight_files:
        print(" -", p, flush=True)

    safetensor_files = [p for p in weight_files if p.endswith(".safetensors")]
    bin_files = [p for p in weight_files if p.endswith(".bin")]

    if safetensor_files:
        weight_path = safetensor_files[0]
        print("Using safetensors:", weight_path, flush=True)

        from safetensors.torch import load_file
        state_dict = load_file(weight_path, device="cpu")
    else:
        weight_path = bin_files[0]
        print("Using torch bin:", weight_path, flush=True)

        state_dict = torch.load(weight_path, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = {}

    for k, v in state_dict.items():
        new_k = k

        if new_k.startswith("model."):
            new_k = new_k[len("model."):]

        if new_k.startswith("module."):
            new_k = new_k[len("module."):]

        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print("[OK] Weights loaded.", flush=True)
    print("Missing keys:", len(missing), flush=True)
    print("Unexpected keys:", len(unexpected), flush=True)

    if len(missing) > 0:
        print("First missing keys:", missing[:10], flush=True)

    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10], flush=True)

    model.eval()

    return model, config


# ============================================================
# 6. 模型输出处理
# ============================================================

def get_logits(out):
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    if hasattr(out, "logits"):
        return out.logits
    return out


def to_1d_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.squeeze(0).detach().cpu()
    return x.reshape(-1)


def decode_forecast(
    logits,
    context_end: pd.Timestamp,
    city_tz: str,
    weather_class_names: list[str],
    forecast_hours: int,
) -> list[dict[str, Any]]:

    if not isinstance(logits, (list, tuple)) or len(logits) != 12:
        raise RuntimeError(
            f"Expected logits to be list/tuple with 12 outputs, "
            f"but got {type(logits)}."
        )

    (
        temp_pred,
        humidity_pred,
        apparent_pred,
        precip_pred,
        sea_level_pressure_pred,
        surface_pressure_pred,
        cloud_cover_pred,
        wind_pred,
        wind_dir_sin_pred,
        wind_dir_cos_pred,
        rain_logit,
        weather_logits,
    ) = logits

    temp_pred = to_1d_tensor(temp_pred)
    humidity_pred = to_1d_tensor(humidity_pred)
    apparent_pred = to_1d_tensor(apparent_pred)
    precip_pred = to_1d_tensor(precip_pred)
    sea_level_pressure_pred = to_1d_tensor(sea_level_pressure_pred)
    surface_pressure_pred = to_1d_tensor(surface_pressure_pred)
    cloud_cover_pred = to_1d_tensor(cloud_cover_pred)
    wind_pred = to_1d_tensor(wind_pred)
    wind_dir_sin_pred = to_1d_tensor(wind_dir_sin_pred)
    wind_dir_cos_pred = to_1d_tensor(wind_dir_cos_pred)

    rain_prob = torch.sigmoid(rain_logit).squeeze(0).detach().cpu().reshape(-1)
    weather_probs = torch.softmax(weather_logits, dim=-1).squeeze(0).detach().cpu()
    weather_idx = torch.argmax(weather_probs, dim=-1).reshape(-1)

    humidity_pred = torch.clamp(humidity_pred, 0.0, 100.0)
    cloud_cover_pred = torch.clamp(cloud_cover_pred, 0.0, 100.0)
    precip_pred = torch.clamp(precip_pred, 0.0)
    wind_pred = torch.clamp(wind_pred, 0.0)
    rain_prob = torch.clamp(rain_prob, 0.0, 1.0)

    horizon = min(
        int(forecast_hours),
        temp_pred.shape[0],
        humidity_pred.shape[0],
        apparent_pred.shape[0],
        precip_pred.shape[0],
        weather_idx.shape[0],
    )

    forecast = []

    for lead in range(1, horizon + 1):
        idx = lead - 1
        target_time = context_end + pd.Timedelta(hours=lead)
        w_idx = int(weather_idx[idx])

        forecast.append(
            {
                "lead_hours": lead,
                "target_utc": target_time.isoformat(),
                "target_local": to_iso(target_time, city_tz),
                "temperature_2m_c": float(temp_pred[idx]),
                "relative_humidity_2m_pct": float(humidity_pred[idx]),
                "apparent_temperature_c": float(apparent_pred[idx]),
                "precipitation_mm": float(precip_pred[idx]),
                "pressure_msl_hpa": float(sea_level_pressure_pred[idx]),
                "surface_pressure_hpa": float(surface_pressure_pred[idx]),
                "cloud_cover_pct": float(cloud_cover_pred[idx]),
                "wind_speed_10m_kmh": float(wind_pred[idx]),
                "wind_direction_sin": float(wind_dir_sin_pred[idx]),
                "wind_direction_cos": float(wind_dir_cos_pred[idx]),
                "rain_probability": float(rain_prob[idx]),
                "weather_class": w_idx,
                "weather_class_name": (
                    weather_class_names[w_idx]
                    if w_idx < len(weather_class_names)
                    else f"class_{w_idx}"
                ),
                "weather_class_probabilities": {
                    name: float(prob)
                    for name, prob in zip(weather_class_names, weather_probs[idx])
                },
            }
        )

    return forecast


# ============================================================
# 7. 主推理流程
# ============================================================

def predict(city: str):
    print("=" * 70, flush=True)
    print("Hweh-446k Real Weather Inference Test", flush=True)
    print("=" * 70, flush=True)
    print("Model ID:", MODEL_ID, flush=True)
    print("Local model backup:", LOCAL_MODEL_DIR, flush=True)
    print("City:", city, flush=True)
    print("Torch version:", torch.__version__, flush=True)
    print("CUDA/DCU available:", torch.cuda.is_available(), flush=True)

    device = torch.device(
        DEVICE if DEVICE else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Using device:", device, flush=True)

    model, config = load_model_manual(MODEL_ID)

    if hasattr(config, "seq_len") and int(config.seq_len) != CONTEXT_HOURS:
        raise ValueError(f"Set CONTEXT_HOURS to {int(config.seq_len)} for this model.")

    if city not in CITY_SPECS:
        raise ValueError(f"Unknown city: {city}. Available cities: {list(CITY_SPECS.keys())}")

    city_spec = CITY_SPECS[city]
    city_tz = CITY_TIMEZONES.get(city, "UTC")

    print("\n[Step 5] Fetching real weather data...", flush=True)

    df = fetch_recent_history(city, CONTEXT_HOURS)
    seq = build_single_sequence(df)

    context_start = df["time"].iloc[0]
    context_end = df["time"].iloc[-1]

    print("Weather dataframe shape:", df.shape, flush=True)
    print("Built sequence shape:", seq.shape, flush=True)
    print("Context start:", context_start.isoformat(), flush=True)
    print("Context end:", context_end.isoformat(), flush=True)

    X = torch.from_numpy(seq).unsqueeze(0)

    # 简化处理：这里按 location_id - 1 转为模型内部 index
    # Seattle: 1 -> 0, New York: 9 -> 8, Singapore: 56 -> 55
    model_location_id = int(city_spec["location_id"]) - 1
    loc = torch.tensor([model_location_id], dtype=torch.long)

    model = model.to(device)
    X = X.to(device)
    loc = loc.to(device)

    print("\n[Step 6] Running model inference...", flush=True)
    print("Input X shape:", tuple(X.shape), flush=True)
    print("location_id shape:", tuple(loc.shape), flush=True)
    print("model_location_id:", model_location_id, flush=True)

    infer_start = time.time()

    with torch.no_grad():
        out = model(X=X, location_id=loc)
        logits = get_logits(out)

    infer_time = time.time() - infer_start

    print("Inference finished.", flush=True)
    print("Inference time: {:.6f} s".format(infer_time), flush=True)

    weather_class_names = getattr(config, "weather_class_names", None)
    if not weather_class_names:
        weather_class_names = [
            "clear",
            "cloudy",
            "fog",
            "drizzle",
            "rain",
            "snow",
            "thunderstorm",
        ]

    forecast = decode_forecast(
        logits=logits,
        context_end=context_end,
        city_tz=city_tz,
        weather_class_names=weather_class_names,
        forecast_hours=FORECAST_HOURS,
    )

    requested_at_utc = pd.Timestamp.now(tz="UTC").isoformat()

    result = {
        "city": city,
        "location_id": str(city_spec["location_id"]),
        "model_location_id": int(model_location_id),
        "data_source": "open-meteo forecast api (past-hours context only)",
        "requested_at_utc": requested_at_utc,
        "model": {
            "model_id": MODEL_ID,
            "local_model_backup": LOCAL_MODEL_DIR,
            "encoder_type": getattr(config, "encoder_type", None),
            "seq_len": int(getattr(config, "seq_len", CONTEXT_HOURS)),
            "input_dim": int(getattr(config, "input_dim", seq.shape[1])),
            "num_weather_classes": int(
                getattr(config, "num_weather_classes", len(weather_class_names))
            ),
        },
        "context": {
            "hours": int(len(df)),
            "start_utc": context_start.isoformat(),
            "end_utc": context_end.isoformat(),
            "start_local": to_iso(context_start, city_tz),
            "end_local": to_iso(context_end, city_tz),
        },
        "forecast": forecast,
        "performance": {
            "inference_time_seconds": float(infer_time),
            "device": str(device),
        },
        "sanity": {
            "sequence_shape": list(seq.shape),
            "finite_features": bool(np.isfinite(seq).all()),
        },
    }

    print("\n[Step 7] Forecast result JSON:", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    print("\n[SUCCESS] Real weather inference completed.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--city",
        type=str,
        default="Seattle",
        help="City name. Available: Seattle, New York, Singapore",
    )
    args = parser.parse_args()

    predict(args.city)


if __name__ == "__main__":
    main()