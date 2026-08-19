"""Run Aurora inference from the OneScience ERA5Dataset.

The entry point deliberately keeps data loading, checkpoint loading, model execution and output
writing explicit.  It does not download checkpoints or infer a checkpoint format from a filename.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"
CHANNEL_COUNT = 69
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return result


def resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.resolve().parents[1] / path).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=None, help="OneScience ERA5 directory")
    parser.add_argument("--static-file", type=Path, default=None, help="Aurora static .npz file")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-type",
        choices=("official", "training"),
        default=None,
        help="official Microsoft .ckpt or this project's aurora-training-checkpoint-v1 .pt",
    )
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--forecast-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def choose_device(requested: str):
    import torch

    value = requested.lower()
    if value == "dcu":
        value = "cuda"
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA/DCU device was requested but torch.cuda.is_available() is false")
    return device


def choose_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def decode_collated_times(time_index: Any, batch_size: int, expected_length: int) -> list[list[str]]:
    """Decode ERA5Dataset's default-collate representation to per-sample timestamps."""
    positions = _as_list(time_index)
    if len(positions) != expected_length:
        raise ValueError(
            f"Expected {expected_length} time positions from ERA5Dataset, got {len(positions)}"
        )
    rows: list[list[str]] = [[] for _ in range(batch_size)]
    for position in positions:
        values = _as_list(position)
        if len(values) == 1 and batch_size > 1:
            values *= batch_size
        if len(values) != batch_size:
            raise ValueError(f"Could not decode collated time position {position!r}")
        for sample, value in enumerate(values):
            text = str(value.decode() if isinstance(value, bytes) else value)
            # Fail early rather than producing metadata which cannot be parsed by result.py.
            datetime.strptime(text, "%Y%m%d%H")
            rows[sample].append(text)
    return rows


def unit_map(channels: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for channel in channels:
        if channel in {"2m_temperature"} or channel.startswith("temperature_"):
            result[channel] = "K"
        elif channel == "mean_sea_level_pressure":
            result[channel] = "Pa"
        elif channel.startswith("geopotential_"):
            result[channel] = "m^2 s^-2"
        elif channel.startswith("specific_humidity_"):
            result[channel] = "kg/kg"
        elif "wind" in channel:
            result[channel] = "m/s"
        else:
            result[channel] = "unknown"
    return result


def normalise_output_steps(outvar, forecast_steps: int):
    import torch

    if outvar.ndim == 4:
        outvar = outvar.unsqueeze(1)
    if outvar.ndim != 5 or outvar.shape[1] != forecast_steps:
        raise ValueError(
            f"ERA5Dataset target must be [B,{forecast_steps},C,H,W], got {tuple(outvar.shape)}"
        )
    return outvar


def load_checkpoint(model, path: Path, checkpoint_type: str) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if checkpoint_type == "official":
        model.load_checkpoint_local(path, strict=True)
        return {"schema": "microsoft-aurora-local-checkpoint", "path": str(path)}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != "aurora-training-checkpoint-v1":
        raise ValueError(
            "--checkpoint-type training requires schema_version="
            "aurora-training-checkpoint-v1 from scripts/train.py"
        )
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Project training checkpoint has no model_state mapping")
    model.load_state_dict(state, strict=True)
    return {
        "schema": payload["schema_version"],
        "path": str(path),
        "training_step": int(payload.get("step", -1)),
        "training_mode": payload.get("mode"),
    }


def setup_output_dir(path: Path, overwrite: bool) -> None:
    existing = [path / name for name in ("prediction.npy", "truth.npy", "input.npy", "metadata.json")]
    if not overwrite and any(item.exists() for item in existing):
        raise FileExistsError(
            f"Inference output already exists at {path}; choose another --output-dir or pass --overwrite"
        )
    path.mkdir(parents=True, exist_ok=True)


def run_inference(args: argparse.Namespace) -> Path:
    import torch
    from torch.utils.data import DataLoader

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    infer_cfg = cfg.get("inference", {})
    channels = list(data_cfg["channel_order"])
    if len(channels) != CHANNEL_COUNT:
        raise ValueError(f"Aurora base inference requires exactly {CHANNEL_COUNT} channels")
    input_steps = int(data_cfg["input_steps"])
    forecast_steps = int(
        args.forecast_steps if args.forecast_steps is not None else infer_cfg.get("forecast_steps", 1)
    )
    if forecast_steps < 1:
        raise ValueError("--forecast-steps must be positive")
    years = list(args.years or infer_cfg.get("years") or data_cfg["test_years"])
    data_dir = resolve_path(args.data_dir or data_cfg["virtual_dir"], config_path)
    if args.static_file is not None:
        static_file = args.static_file.resolve()
    elif args.data_dir is not None:
        static_file = (data_dir / "static" / "static_vars.npz").resolve()
    else:
        static_file = resolve_path(data_cfg["static_file"], config_path)
    checkpoint_type = args.checkpoint_type or str(infer_cfg.get("checkpoint_type", "official"))
    if checkpoint_type not in {"official", "training"}:
        raise ValueError("checkpoint_type must be 'official' or 'training'")
    checkpoint_value = args.checkpoint or infer_cfg.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("An explicit --checkpoint or inference.checkpoint is required")
    checkpoint = resolve_path(checkpoint_value, config_path)
    output_dir = resolve_path(
        args.output_dir or infer_cfg.get("output_dir", "outputs/inference/aurora"), config_path
    )
    batch_size = int(args.batch_size if args.batch_size is not None else infer_cfg.get("batch_size", 1))
    num_workers = int(args.num_workers if args.num_workers is not None else infer_cfg.get("num_workers", 0))
    max_samples_value = args.max_samples if args.max_samples is not None else infer_cfg.get("max_samples")
    max_samples = None if max_samples_value in (None, 0) else int(max_samples_value)
    if batch_size < 1 or num_workers < 0 or (max_samples is not None and max_samples < 1):
        raise ValueError("batch size and max samples must be positive; workers cannot be negative")
    requested_device = str(args.device or infer_cfg.get("device", cfg.get("runtime", {}).get("device", "auto")))
    dtype_name = str(args.dtype or infer_cfg.get("dtype", "float32"))
    device = choose_device(requested_device)
    dtype = choose_dtype(dtype_name)
    seed_everything(int(args.seed if args.seed is not None else cfg["project"]["seed"]))
    setup_output_dir(output_dir, args.overwrite)

    from onescience.datapipes.climate import ERA5Dataset
    from model.aurora import build_aurora_model

    dataset = ERA5Dataset(
        dataset_dir=str(data_dir),
        used_years=years,
        used_variables=channels,
        mode="test",
        input_steps=input_steps,
        output_steps=forecast_steps,
        normalize=bool(data_cfg["normalize_in_onescience"]),
    )
    total_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    if total_samples < 1:
        raise ValueError("ERA5Dataset contains no samples for the selected years")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model_cfg_for_build = dict(cfg)
    model_cfg_for_build["data"] = dict(data_cfg)
    model_cfg_for_build["data"]["static_file"] = str(static_file)
    model = build_aurora_model(model_cfg_for_build, project_root=PROJECT_ROOT, load_pretrained=False)
    checkpoint_info = load_checkpoint(model, checkpoint, checkpoint_type)
    model.to(device=device, dtype=dtype)
    model.eval()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("checkpoint=%s type=%s device=%s dtype=%s", checkpoint, checkpoint_type, device, dtype_name)

    with np.load(static_file) as static:
        lat_input = np.asarray(static["lat"], dtype=np.float32)
        lon = np.asarray(static["lon"], dtype=np.float32)
    expected_height = int(data_cfg["grid"]["virtual_height"])
    expected_width = int(data_cfg["grid"]["virtual_width"])
    if lat_input.shape != (expected_height,) or lon.shape != (expected_width,):
        raise ValueError("Static coordinate shapes do not match the configured ERA5 grid")

    prediction_mm = truth_mm = input_mm = None
    init_times: list[str] = []
    valid_times: list[list[str]] = []
    processed = 0
    prediction_min = float("inf")
    prediction_max = float("-inf")
    with torch.inference_mode():
        for invar, outvar, _, _, time_index in loader:
            if processed >= total_samples:
                break
            if invar.ndim == 4:
                invar = invar.unsqueeze(1)
            if invar.ndim != 5 or invar.shape[1] != input_steps or invar.shape[2] != CHANNEL_COUNT:
                raise ValueError(f"ERA5Dataset input has unexpected shape {tuple(invar.shape)}")
            outvar = normalise_output_steps(outvar, forecast_steps)
            batch_size_actual = invar.shape[0]
            times = decode_collated_times(time_index, batch_size_actual, input_steps + forecast_steps)
            take = min(batch_size_actual, total_samples - processed)
            invar = invar[:take].to(device=device, dtype=torch.float32, non_blocking=True)
            outvar = outvar[:take].to(device=device, dtype=torch.float32, non_blocking=True)
            init_batch_times = [row[input_steps - 1] for row in times[:take]]
            valid_batch_times = [row[input_steps:] for row in times[:take]]
            if forecast_steps == 1:
                pred = model(invar, times=init_batch_times).unsqueeze(1)
            else:
                pred = model.rollout(invar, times=init_batch_times, steps=forecast_steps)
            target = model.crop_target(outvar, int(model_cfg["patch_size"]))
            if pred.ndim != 5 or tuple(pred.shape) != tuple(target.shape):
                raise ValueError(
                    f"Prediction/target shape mismatch: prediction={tuple(pred.shape)} target={tuple(target.shape)}"
                )
            if not torch.isfinite(pred).all().item() or not torch.isfinite(target).all().item():
                raise ValueError("Aurora produced a non-finite prediction or target")
            pred_np = pred.float().cpu().numpy()
            target_np = target.float().cpu().numpy()
            input_np = invar.float().cpu().numpy()
            if prediction_mm is None:
                prediction_mm = np.lib.format.open_memmap(
                    output_dir / "prediction.npy", mode="w+", dtype=np.float32,
                    shape=(total_samples, *pred_np.shape[1:]),
                )
                truth_mm = np.lib.format.open_memmap(
                    output_dir / "truth.npy", mode="w+", dtype=np.float32,
                    shape=(total_samples, *target_np.shape[1:]),
                )
                input_mm = np.lib.format.open_memmap(
                    output_dir / "input.npy", mode="w+", dtype=np.float32,
                    shape=(total_samples, *input_np.shape[1:]),
                )
            prediction_mm[processed : processed + take] = pred_np
            truth_mm[processed : processed + take] = target_np
            input_mm[processed : processed + take] = input_np
            prediction_min = min(prediction_min, float(pred_np.min()))
            prediction_max = max(prediction_max, float(pred_np.max()))
            init_times.extend(init_batch_times)
            valid_times.extend(valid_batch_times)
            processed += take
    if processed != total_samples or prediction_mm is None or truth_mm is None or input_mm is None:
        raise RuntimeError(f"Inference wrote {processed} samples, expected {total_samples}")
    prediction_mm.flush()
    truth_mm.flush()
    input_mm.flush()
    prediction_shape = list(prediction_mm.shape)
    truth_shape = list(truth_mm.shape)
    input_shape = list(input_mm.shape)
    if prediction_shape[-2] == lat_input.size:
        lat = lat_input
    elif prediction_shape[-2] == lat_input.size - 1:
        lat = lat_input[:-1]
    else:
        raise ValueError("Aurora prediction latitude shape is incompatible with the static grid")
    np.save(output_dir / "lat.npy", lat)
    np.save(output_dir / "lon.npy", lon)
    metadata = {
        "schema_version": "aurora-inference-output-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "variant": str(model_cfg.get("variant", "small")),
            "official_class": str(model_cfg.get("official_class", "AuroraSmallPretrained")),
            "checkpoint": checkpoint_info,
        },
        "loader": {
            "name": "onescience.datapipes.climate.ERA5Dataset",
            "dataset_dir": str(data_dir),
            "years": years,
            "normalize": bool(data_cfg["normalize_in_onescience"]),
        },
        "arrays": {
            "prediction": {"path": "prediction.npy", "shape": prediction_shape, "dtype": "float32"},
            "truth": {"path": "truth.npy", "shape": truth_shape, "dtype": "float32"},
            "input": {"path": "input.npy", "shape": input_shape, "dtype": "float32"},
            "lat": {"path": "lat.npy", "shape": list(lat.shape), "dtype": "float32"},
            "lon": {"path": "lon.npy", "shape": list(lon.shape), "dtype": "float32"},
        },
        "channel_order": channels,
        "units": unit_map(channels),
        "surface_vars": list(model_cfg["surface_vars"]),
        "static_vars": list(model_cfg["static_vars"]),
        "atmos_vars": list(model_cfg["atmos_vars"]),
        "atmos_levels_hpa": list(model_cfg["atmos_levels"]),
        "history_steps": input_steps,
        "forecast_steps": forecast_steps,
        "timestep_hours": int(data_cfg["time_step_hours"]),
        "init_times_utc": init_times,
        "valid_times_utc": valid_times,
        "lead_times_hours": [int(data_cfg["time_step_hours"]) * (step + 1) for step in range(forecast_steps)],
        "latitude_order": "north_to_south",
        "longitude_convention": "0_to_360",
        "official_patch_crop": {"input_height": int(input_shape[-2]), "output_height": int(prediction_shape[-2])},
        "normalization": {
            "onescience": "disabled",
            "aurora": "official internal normalization and unnormalization",
        },
        "prediction_range": {"min": prediction_min, "max": prediction_max},
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "aurora-inference-summary-v1",
                "status": "completed",
                "samples": total_samples,
                "forecast_steps": forecast_steps,
                "output_dir": str(output_dir),
                "prediction_shape": prediction_shape,
                "truth_shape": truth_shape,
            },
            handle,
            indent=2,
        )
    logging.info("wrote %d samples to %s", total_samples, output_dir)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_inference(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
