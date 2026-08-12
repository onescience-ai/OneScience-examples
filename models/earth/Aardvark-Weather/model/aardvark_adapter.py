"""Thin, model-local wrapper for the official Aardvark Weather model."""

from __future__ import annotations

import hashlib
import importlib
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


TOP_LEVEL_KEYS = {"assimilation", "forecast", "downscaling", "y_target"}
REQUIRED_ASSIMILATION_KEYS = {
    "x_context_hadisd_current", "y_context_hadisd_current", "climatology_current",
    "sat_x_current", "sat_current", "icoads_x_current", "icoads_current",
    "igra_x_current", "igra_current", "amsua_current", "amsua_x_current",
    "amsub_current", "amsub_x_current", "iasi_current", "iasi_x_current",
    "ascat_current", "ascat_x_current", "hirs_current", "hirs_x_current",
    "y_target_current", "era5_x_current",
    "era5_elev_current", "era5_lonlat_current", "aux_time_current", "lt",
    "y_target",
}
REQUIRED_FORECAST_KEYS = {"y_context", "y_target", "lt"}
REQUIRED_DOWNSCALING_KEYS = {
    "x_target", "alt_target", "y_target", "y_context", "x_context", "aux_time", "lt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample(sample_path: Path) -> dict[str, Any]:
    with sample_path.open("rb") as handle:
        sample = pickle.load(handle)
    if not isinstance(sample, dict) or set(sample) != TOP_LEVEL_KEYS:
        raise ValueError(f"sample top-level keys mismatch: {list(sample) if isinstance(sample, dict) else type(sample)}")
    expected = {
        "assimilation": REQUIRED_ASSIMILATION_KEYS,
        "forecast": REQUIRED_FORECAST_KEYS,
        "downscaling": REQUIRED_DOWNSCALING_KEYS,
    }
    for name, keys in expected.items():
        if not isinstance(sample[name], dict) or set(sample[name]) != keys:
            raise ValueError(f"sample {name} keys mismatch: {list(sample[name])}")
    if not isinstance(sample["y_target"], torch.Tensor) or sample["y_target"].ndim != 2:
        raise ValueError("sample y_target must be a rank-2 torch.Tensor")
    return {
        "top_level_keys": sorted(sample),
        "nested_keys": {name: sorted(value) for name, value in expected.items()},
        "y_target_shape": list(sample["y_target"].shape),
        "nan_counts": {
            name: int(value.isnan().sum())
            for name, value in sample["downscaling"].items()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        },
    }


def validate_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint contract mismatch: {path}")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError(f"empty model_state_dict: {path}")
    return {"path": str(path), "key_count": len(state), "has_model_state_dict": True}


def load_sample(sample_path: Path) -> dict[str, Any]:
    with sample_path.open("rb") as handle:
        return pickle.load(handle)


def build_one_day_model(weights_root: Path, official_root: Path, device: str):
    encoder = weights_root / "trained_model/encoder"
    processor = weights_root / "trained_model/processor"
    decoder = weights_root / "trained_model/decoder/tas"
    sys.path.insert(0, str(official_root / "aardvark"))
    _install_timm_compatibility()
    official_e2e = importlib.import_module("e2e_model")
    caller_dir = Path.cwd()
    try:
        os.chdir(official_root / "aardvark")
        model = official_e2e.ConvCNPWeatherE2E(
            device=device,
            lead_time=1,
            se_model_path=str(encoder),
            forecast_model_path=str(processor),
            sf_model_path=str(decoder) + "/",
            return_gridded=True,
            aux_data_path=str(official_root / "data") + "/",
        )
    finally:
        os.chdir(caller_dir)
    return model


def run_one_day(sample_path: Path, weights_root: Path, official_root: Path, device: str) -> dict[str, Any]:
    sample_report = validate_sample(sample_path)
    encoder = weights_root / "trained_model/encoder"
    processor = weights_root / "trained_model/processor"
    decoder = weights_root / "trained_model/decoder/tas"
    checkpoint_report = [
        validate_checkpoint(encoder / "epoch_96"),
        validate_checkpoint(processor / "forecast_1/epoch_0"),
        validate_checkpoint(decoder / "lt_1/epoch_18"),
    ]
    sample = load_sample(sample_path)
    model = build_one_day_model(weights_root, official_root, device)
    model.eval()
    with torch.inference_mode():
        station, global_forecast, initial_state = model(sample)
    for name, tensor in (("station_tas", station), ("global_forecast", global_forecast), ("initial_state", initial_state)):
        if not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values")
    return {
        "sample": sample_report,
        "checkpoints": checkpoint_report,
        "device": device,
        "lead_time_days": 1,
        "station_tas_shape": list(station.shape),
        "global_forecast_shape": list(global_forecast.shape),
        "initial_state_shape": list(initial_state.shape),
        "finite_outputs": True,
    }


def _install_timm_compatibility() -> None:
    """Bridge the old timm 0.6 Block constructor used by the official code."""
    import timm.models.vision_transformer as vision_transformer

    original = vision_transformer.Block
    if getattr(original, "_aardvark_compat", False):
        return

    class AardvarkBlock(original):
        _aardvark_compat = True

        def __init__(self, *args: Any, drop: float = 0.0, **kwargs: Any) -> None:
            kwargs.setdefault("proj_drop", drop)
            super().__init__(*args, **kwargs)

    vision_transformer.Block = AardvarkBlock
