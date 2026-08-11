#!/usr/bin/env python3
"""Validate ECMWF AIFS Single v1.0 with the official Anemoi runner.

Despite the historical filename ``train.py``, this script does not retrain
AIFS. It verifies the official checkpoint, constructs or loads the two input
states required by AIFS, runs one or more forecasts, calculates functional
metrics, and writes reproducible validation artifacts.

Default functional validation (uses local ``input_state.npz`` when present,
otherwise creates a deterministic synthetic N320-like state)::

    python train.py --device cuda

Use current ECMWF Open Data as initial conditions::

    python train.py --device cuda --data-source open-data

Use a locally prepared NPZ initial state::

    python train.py --device cuda --input input_state.npz

The synthetic input is deliberately out-of-distribution. It can demonstrate
that the published checkpoint loads and produces finite, correctly structured
outputs, but it cannot measure scientific forecast skill.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


MODEL_NAME = "ecmwf/aifs-single-1.0"
CHECKPOINT_NAME = "aifs-single-mse-1.0.ckpt"
EXPECTED_CHECKPOINT_BYTES = 994_084_883
EXPECTED_CHECKPOINT_SHA256 = (
    "1fed399c097c0127d5bbe074f4f8bbc123759736145d990699c215ff07543ccd"
)
GRID_POINTS = 542_080
PRESSURE_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
SURFACE_INPUT_FIELDS = (
    "10u",
    "10v",
    "2d",
    "2t",
    "msl",
    "skt",
    "sp",
    "tcw",
    "lsm",
    "z",
    "slor",
    "sdor",
    "stl1",
    "stl2",
    "swvl1",
    "swvl2",
)
PRESSURE_VARIABLES = ("z", "t", "u", "v", "w", "q")
STANDARD_TEMPERATURE_BY_LEVEL = {
    1000: 288.0,
    925: 283.0,
    850: 278.0,
    700: 268.0,
    600: 260.0,
    500: 252.0,
    400: 244.0,
    300: 232.0,
    250: 224.0,
    200: 216.0,
    150: 216.0,
    100: 216.0,
    50: 225.0,
}
PRESSURE_INPUT_FIELDS = tuple(
    f"{variable}_{level}"
    for variable in PRESSURE_VARIABLES
    for level in PRESSURE_LEVELS
)
INPUT_FIELDS = SURFACE_INPUT_FIELDS + PRESSURE_INPUT_FIELDS
SELECTED_OUTPUT_FIELDS = ("2t", "msl", "t_500", "z_500", "10u", "10v", "tp", "tcc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Functionally validate the official ECMWF AIFS Single v1.0 "
            "checkpoint and save machine-readable results."
        )
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device. CUDA is strongly recommended and required by the official FlashAttention setup.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing the checkpoint and optional input_state.npz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Checkpoint path (default: <model-dir>/{CHECKPOINT_NAME}).",
    )
    parser.add_argument(
        "--data-source",
        choices=("auto", "synthetic", "npz", "open-data"),
        default="auto",
        help=(
            "auto uses a local NPZ when found and otherwise generates synthetic data; "
            "open-data follows the official ECMWF example and requires internet access."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="NPZ initial state. Arrays must be named with the official field names and have shape (2, 542080).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Optional NPZ reference at the final lead time for paired MAE/RMSE calculation.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Initial time in ISO format. Used by synthetic/NPZ input; open-data defaults to the latest cycle.",
    )
    parser.add_argument(
        "--lead-time",
        type=int,
        default=6,
        help="Forecast lead time in hours; AIFS advances in 6-hour steps (default: 6).",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=16,
        help="ANEMOI_INFERENCE_NUM_CHUNKS value used to reduce mapper memory (default: 16).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Untimed forecast runs before measurement (default: 0 because a global forecast is expensive).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of timed forecasts (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result folder (default: <model-dir>/validation_results).",
    )
    parser.add_argument(
        "--save-full-input",
        action="store_true",
        help="Save the complete generated/downloaded input NPZ (roughly 400 MB before compression).",
    )
    parser.add_argument(
        "--save-full-output",
        action="store_true",
        help="Save all forecast fields at the final lead time.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only verify the local checkpoint, Python, PyTorch and CUDA; do not import Anemoi or run inference.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Allow an unverified checkpoint. Unsafe because the official CKPT is a pickle file.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path, skip_checksum: bool) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Expected the official file {CHECKPOINT_NAME} in the model folder."
        )
    file_bytes = path.stat().st_size
    if skip_checksum:
        print(
            "WARNING: checkpoint checksum verification was disabled. Loading an "
            "unverified pickle checkpoint can execute arbitrary code.",
            file=sys.stderr,
        )
        return {
            "path": str(path),
            "bytes": file_bytes,
            "sha256": None,
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
            "checksum_verified": False,
            "size_matches_official": file_bytes == EXPECTED_CHECKPOINT_BYTES,
        }
    actual = sha256_file(path)
    if file_bytes != EXPECTED_CHECKPOINT_BYTES or actual != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint verification failed. Refusing to unpickle it.\n"
            f"Path: {path}\n"
            f"Bytes: {file_bytes} (expected {EXPECTED_CHECKPOINT_BYTES})\n"
            f"SHA256: {actual}\n"
            f"Expected: {EXPECTED_CHECKPOINT_SHA256}"
        )
    return {
        "path": str(path),
        "bytes": file_bytes,
        "sha256": actual,
        "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checksum_verified": True,
        "size_matches_official": True,
    }


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_audit() -> Dict[str, Dict[str, Any]]:
    """Import required/optional packages without installing or changing them."""

    specifications = (
        ("anemoi.inference", "anemoi-inference", True, "official model runner"),
        ("anemoi.models", "anemoi-models", True, "checkpoint model classes"),
        ("anemoi.transform", "anemoi-transform", True, "Anemoi preprocessing"),
        ("anemoi.utils", "anemoi-utils", True, "Anemoi runtime utilities"),
        ("flash_attn", "flash-attn", True, "checkpoint CUDA attention kernel"),
        ("torch_geometric", "torch-geometric", True, "checkpoint graph layers"),
        ("omegaconf", "omegaconf", True, "checkpoint configuration objects"),
        ("hydra", "hydra-core", True, "Anemoi model configuration"),
        ("einops", "einops", True, "Anemoi tensor transformations"),
        ("earthkit.data", "earthkit-data", True, "Anemoi state interface"),
        ("eccodes", "eccodes", True, "Anemoi inference dependency"),
        ("aniso8601", "aniso8601", True, "Anemoi lead-time parsing"),
        ("semantic_version", "semantic-version", True, "Anemoi version checks"),
        ("anytree", "anytree", True, "Anemoi metadata structures"),
        ("matplotlib", "matplotlib", False, "preview image"),
        ("earthkit.regrid", "earthkit-regrid", False, "ECMWF open-data mode"),
        ("ecmwf.opendata", "ecmwf-opendata", False, "ECMWF open-data mode"),
    )
    report: Dict[str, Dict[str, Any]] = {}
    for module_name, distribution_name, required, purpose in specifications:
        try:
            imported = importlib.import_module(module_name)
            version = package_version(distribution_name) or getattr(
                imported, "__version__", None
            )
            report[module_name] = {
                "required_for_default_inference": required,
                "purpose": purpose,
                "import_ok": True,
                "version": str(version) if version is not None else "unknown",
                "error": None,
            }
        except Exception as error:
            report[module_name] = {
                "required_for_default_inference": required,
                "purpose": purpose,
                "import_ok": False,
                "version": package_version(distribution_name),
                "error": f"{type(error).__name__}: {error}",
            }
    return report


def install_torch_geometric_checkpoint_compatibility() -> Dict[str, Any]:
    """Restore the legacy PyG Inspector referenced by the official CKPT.

    AIFS Single v1.0 was serialized with PyG 2.4, where ``Inspector`` lived at
    ``torch_geometric.nn.conv.utils.inspector`` and stored ``base_class`` plus
    ``params``. PyG 2.7 moved it to ``torch_geometric.inspector`` and stores
    ``_cls``, ``_signature_dict`` and ``_source_dict`` instead. The in-memory
    shim below resolves the old module path and rebuilds the new Inspector state
    from the message-passing class by following PyG 2.7's own initialization
    sequence. It does not change model weights, PyG files, or inference math.
    """

    legacy_name = "torch_geometric.nn.conv.utils.inspector"
    modern_name = "torch_geometric.inspector"
    try:
        legacy_module = importlib.import_module(legacy_name)
        return {
            "status": "native",
            "legacy_module": legacy_name,
            "target_module": legacy_module.__name__,
            "applied": False,
        }
    except ModuleNotFoundError as error:
        if error.name != legacy_name:
            return {
                "status": "unavailable",
                "legacy_module": legacy_name,
                "target_module": modern_name,
                "applied": False,
                "error": f"{type(error).__name__}: {error}",
            }
    try:
        import types

        modern_module = importlib.import_module(modern_name)
        modern_inspector = getattr(modern_module, "Inspector")

        class LegacyInspectorCompat(modern_inspector):
            """Convert a pickled PyG 2.4 Inspector into PyG 2.7 state."""

            def __setstate__(self, state: Dict[str, Any]) -> None:
                base_class = state.get("base_class")
                if base_class is None:
                    raise RuntimeError(
                        "The legacy PyG Inspector has no base_class; refusing "
                        "to guess its message-passing signatures."
                    )

                modern_inspector.__init__(self, base_class.__class__)
                self.inspect_signature(base_class.message)
                self.inspect_signature(
                    base_class.aggregate,
                    exclude=[0, "aggr"],
                )
                self.inspect_signature(
                    base_class.message_and_aggregate,
                    exclude=[0],
                )
                self.inspect_signature(base_class.update, exclude=[0])
                self.inspect_signature(base_class.edge_update)

                # Keep the legacy attributes for audit/debugging only. PyG 2.7
                # uses the freshly rebuilt modern fields for execution.
                self.base_class = base_class
                self.params = state.get("params", {})

        # Pickle resolves the class by this exact legacy global name.
        LegacyInspectorCompat.__name__ = "Inspector"
        LegacyInspectorCompat.__qualname__ = "Inspector"
        LegacyInspectorCompat.__module__ = legacy_name

        shim = types.ModuleType(legacy_name)
        shim.Inspector = LegacyInspectorCompat
        shim.__all__ = ["Inspector"]
        shim.__doc__ = (
            "Runtime Inspector state adapter for the checksum-verified "
            "ECMWF AIFS Single v1.0 checkpoint."
        )
        sys.modules[legacy_name] = shim
        parent_module = importlib.import_module("torch_geometric.nn.conv.utils")
        setattr(parent_module, "inspector", shim)
        return {
            "status": "legacy_state_adapter_installed",
            "legacy_module": legacy_name,
            "target_module": modern_name,
            "target_class": (
                f"{modern_inspector.__module__}.{modern_inspector.__name__}"
            ),
            "adapter": (
                "PyG 2.4 base_class/params -> PyG 2.7 class signatures"
            ),
            "applied": True,
        }
    except Exception as error:
        return {
            "status": "unavailable",
            "legacy_module": legacy_name,
            "target_module": modern_name,
            "applied": False,
            "error": f"{type(error).__name__}: {error}",
        }


def parse_date(value: Optional[str], default: dt.datetime) -> dt.datetime:
    if value is None:
        return default
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def two_times(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)]
    ).astype(np.float32, copy=False)


def make_synthetic_state(
    seed: int, date: dt.datetime
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create the complete two-time N320-shaped input expected by AIFS.

    Values are smooth and approximately meteorological, but the reduced-Gaussian
    node ordering is only represented by deterministic proxy waves. This state
    is intended exclusively for end-to-end functional validation.
    """

    n = GRID_POINTS
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 8.0 * np.pi, n, endpoint=False, dtype=np.float32)
    lat_proxy = np.sin(phase * 0.25).astype(np.float32)
    wave_a = np.sin(phase).astype(np.float32)
    wave_b = np.cos(phase * 0.71).astype(np.float32)
    small_noise = rng.normal(0.0, 0.05, size=n).astype(np.float32)

    fields: Dict[str, np.ndarray] = {}
    t2_now = 286.0 - 24.0 * np.abs(lat_proxy) + 2.5 * wave_b + small_noise
    t2_previous = t2_now - 0.8 * wave_a
    orography_m = np.maximum(0.0, 950.0 * (wave_a - 0.42)).astype(np.float32)
    land = (wave_b + 0.18 * wave_a > -0.05).astype(np.float32)
    msl_now = 101_325.0 + 1_450.0 * wave_a + 380.0 * wave_b
    msl_previous = msl_now - 160.0 * wave_b
    sp_now = msl_now * np.exp(-orography_m / 8_400.0)
    sp_previous = msl_previous * np.exp(-orography_m / 8_400.0)

    fields["10u"] = two_times(7.0 * wave_a, 7.0 * wave_a + 0.7 * wave_b)
    fields["10v"] = two_times(5.0 * wave_b, 5.0 * wave_b - 0.5 * wave_a)
    fields["2t"] = two_times(t2_previous, t2_now)
    fields["2d"] = two_times(
        t2_previous - 3.5 - np.abs(wave_b),
        t2_now - 3.5 - np.abs(wave_b),
    )
    fields["msl"] = two_times(msl_previous, msl_now)
    fields["skt"] = two_times(t2_previous + 1.0, t2_now + 1.2)
    fields["sp"] = two_times(sp_previous, sp_now)
    fields["tcw"] = two_times(
        24.0 + 14.0 * (1.0 - np.abs(lat_proxy)),
        24.5 + 14.0 * (1.0 - np.abs(lat_proxy)),
    )
    fields["lsm"] = two_times(land, land)
    fields["z"] = two_times(orography_m * 9.80665, orography_m * 9.80665)
    fields["slor"] = two_times(
        0.03 + 0.16 * np.abs(wave_a),
        0.03 + 0.16 * np.abs(wave_a),
    )
    fields["sdor"] = two_times(
        12.0 + 110.0 * np.maximum(wave_a, 0.0),
        12.0 + 110.0 * np.maximum(wave_a, 0.0),
    )
    fields["stl1"] = two_times(t2_previous - 0.5, t2_now - 0.3)
    fields["stl2"] = two_times(t2_previous - 1.5, t2_now - 1.3)
    soil_moisture = np.clip(
        0.24 + 0.12 * wave_b + 0.04 * lat_proxy, 0.02, 0.60
    )
    fields["swvl1"] = two_times(soil_moisture, soil_moisture + 0.002 * wave_a)
    fields["swvl2"] = two_times(
        np.clip(soil_moisture + 0.03, 0.02, 0.65),
        np.clip(soil_moisture + 0.032, 0.02, 0.65),
    )

    for level in PRESSURE_LEVELS:
        pressure_ratio = np.float32(level / 1000.0)
        height_m = np.float32(
            44_330.0 * (1.0 - (level / 1013.25) ** 0.1903)
        )
        temperature_mean = np.float32(STANDARD_TEMPERATURE_BY_LEVEL[level])
        temperature_now = (
            temperature_mean
            - (7.0 + 8.0 * pressure_ratio) * np.abs(lat_proxy)
            + 1.5 * wave_b
        )
        temperature_previous = temperature_now - 0.35 * wave_a
        wind_scale = np.float32(7.0 + 18.0 * (1.0 - pressure_ratio))
        u_now = wind_scale * wave_a + 2.0 * wave_b
        v_now = 0.75 * wind_scale * wave_b - 1.5 * wave_a
        humidity_now = np.maximum(
            1.0e-6,
            0.012 * pressure_ratio**3 * (0.55 + 0.45 * (1.0 - np.abs(lat_proxy))),
        )
        vertical_now = 0.018 * wave_a * wave_b * pressure_ratio
        geopotential = (
            height_m * 9.80665
            + 180.0 * wave_b * (1.0 - pressure_ratio)
        )
        fields[f"z_{level}"] = two_times(
            geopotential - 8.0 * wave_a, geopotential
        )
        fields[f"t_{level}"] = two_times(
            temperature_previous, temperature_now
        )
        fields[f"u_{level}"] = two_times(u_now - 0.8 * wave_b, u_now)
        fields[f"v_{level}"] = two_times(v_now + 0.6 * wave_a, v_now)
        fields[f"w_{level}"] = two_times(
            vertical_now - 0.002 * wave_b, vertical_now
        )
        fields[f"q_{level}"] = two_times(
            humidity_now * (1.0 - 0.015 * wave_a), humidity_now
        )

    validate_input_fields(fields)
    state = {"date": date, "fields": fields}
    metadata = {
        "kind": "deterministic_synthetic_n320_like",
        "seed": seed,
        "date": date.isoformat(),
        "time_steps": 2,
        "time_spacing_hours": 6,
        "field_count": len(fields),
        "grid_points": GRID_POINTS,
        "full_state_generated_in_memory": True,
        "interpretation": (
            "Out-of-distribution synthetic input for functional validation only; "
            "not a substitute for ECMWF analysis initial conditions."
        ),
    }
    return state, metadata


def validate_input_fields(fields: Mapping[str, np.ndarray]) -> None:
    missing = [name for name in INPUT_FIELDS if name not in fields]
    if missing:
        raise ValueError(
            f"Input state is missing {len(missing)} required fields: {missing[:12]}"
        )
    problems = []
    for name in INPUT_FIELDS:
        array = np.asarray(fields[name])
        if array.shape != (2, GRID_POINTS):
            problems.append(f"{name}: {array.shape}")
        if not np.isfinite(array).all():
            problems.append(f"{name}: contains non-finite values")
        if len(problems) >= 12:
            break
    if problems:
        raise ValueError(
            "Input fields must be finite float arrays with shape "
            f"(2, {GRID_POINTS}). Problems: {problems}"
        )


def load_npz_state(
    path: Path, date_override: Optional[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {path}")
    fields: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        archive_names = set(archive.files)
        if date_override:
            date = parse_date(date_override, dt.datetime(2024, 1, 1))
        elif "date" in archive_names:
            date_value = np.asarray(archive["date"]).reshape(-1)[0]
            date = parse_date(str(date_value), dt.datetime(2024, 1, 1))
        else:
            date = dt.datetime(2024, 1, 1)
        for name in INPUT_FIELDS:
            key = name if name in archive_names else f"field__{name}"
            if key in archive_names:
                fields[name] = np.asarray(archive[key], dtype=np.float32)
    validate_input_fields(fields)
    return (
        {"date": date, "fields": fields},
        {
            "kind": "local_npz",
            "path": str(path),
            "date": date.isoformat(),
            "time_steps": 2,
            "field_count": len(fields),
            "grid_points": GRID_POINTS,
        },
    )


def fetch_open_data_state(
    date_override: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Follow the model-card notebook to construct real ECMWF initial fields."""

    try:
        import earthkit.data as ekd
        import earthkit.regrid as ekr
        from ecmwf.opendata import Client as OpendataClient
    except ImportError as error:
        raise RuntimeError(
            "Open-data mode requires earthkit-data, earthkit-regrid==0.4.0 "
            "and ecmwf-opendata. Check the existing container first and obtain "
            "administrator approval before installing missing packages."
        ) from error

    if date_override:
        date = parse_date(date_override, dt.datetime.utcnow())
    else:
        date = OpendataClient().latest()
        if date.tzinfo is not None:
            date = date.astimezone(dt.timezone.utc).replace(tzinfo=None)

    def get_open_data(
        parameters: Sequence[str], levels: Optional[Sequence[int]] = None
    ) -> Dict[str, np.ndarray]:
        collected: Dict[str, List[np.ndarray]] = {}
        for current_date in (date - dt.timedelta(hours=6), date):
            kwargs: Dict[str, Any] = {
                "date": current_date,
                "param": list(parameters),
            }
            if levels is not None:
                kwargs["levelist"] = list(levels)
            data = ekd.from_source("ecmwf-open-data", **kwargs)
            for field in data:
                values = np.asarray(field.to_numpy())
                if values.shape != (721, 1440):
                    raise ValueError(
                        f"Unexpected ECMWF 0.25-degree field shape {values.shape}."
                    )
                values = np.roll(values, -values.shape[1] // 2, axis=1)
                values = ekr.interpolate(
                    values, {"grid": (0.25, 0.25)}, {"grid": "N320"}
                )
                name = (
                    f"{field.metadata('param')}_{field.metadata('levelist')}"
                    if levels is not None
                    else str(field.metadata("param"))
                )
                collected.setdefault(name, []).append(
                    np.asarray(values, dtype=np.float32)
                )
        return {
            name: np.stack(values).astype(np.float32, copy=False)
            for name, values in collected.items()
        }

    fields: Dict[str, np.ndarray] = {}
    fields.update(
        get_open_data(
            ("10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor")
        )
    )
    soil = get_open_data(("vsw", "sot"), (1, 2))
    soil_mapping = {
        "sot_1": "stl1",
        "sot_2": "stl2",
        "vsw_1": "swvl1",
        "vsw_2": "swvl2",
    }
    for source, destination in soil_mapping.items():
        fields[destination] = soil[source]
    fields.update(
        get_open_data(("gh", "t", "u", "v", "w", "q"), PRESSURE_LEVELS)
    )
    for level in PRESSURE_LEVELS:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * np.float32(9.80665)
    validate_input_fields(fields)
    return (
        {"date": date, "fields": fields},
        {
            "kind": "ecmwf_open_data",
            "date": date.isoformat(),
            "time_steps": 2,
            "field_count": len(fields),
            "grid_points": GRID_POINTS,
            "workflow": "Official model-card 0.25-degree to N320 interpolation",
        },
    )


def find_local_npz(model_dir: Path, output_dir: Path) -> Optional[Path]:
    preferred = model_dir / "input_state.npz"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        path
        for path in model_dir.glob("*.npz")
        if path.is_file() and output_dir not in path.parents
    )
    return candidates[0] if candidates else None


def save_state_npz(path: Path, state: Mapping[str, Any]) -> None:
    payload: Dict[str, Any] = {
        "date": np.asarray(str(state["date"]), dtype="U32")
    }
    payload.update(
        {
            name: np.asarray(values, dtype=np.float32)
            for name, values in state["fields"].items()
        }
    )
    np.savez_compressed(path, **payload)


def save_input_sample(
    path: Path, state: Mapping[str, Any], sample_points: int = 2048
) -> None:
    indexes = np.linspace(
        0, GRID_POINTS - 1, min(sample_points, GRID_POINTS), dtype=np.int64
    )
    payload: Dict[str, Any] = {
        "date": np.asarray(str(state["date"]), dtype="U32"),
        "point_indexes": indexes,
        "note": np.asarray(
            "Audit sample only; reconstruct the full synthetic input with the recorded seed.",
            dtype="U128",
        ),
    }
    for name in INPUT_FIELDS:
        payload[name] = np.asarray(state["fields"][name])[:, indexes]
    np.savez_compressed(path, **payload)


def import_simple_runner() -> Any:
    try:
        from anemoi.inference.runners.simple import SimpleRunner
    except Exception as error:
        raise RuntimeError(
            "Could not import the official Anemoi SimpleRunner. Run "
            "`python train.py --device cuda --preflight-only` and send the "
            "reported missing/failed package names to the administrator. "
            "Do not replace the container's PyTorch/CUDA environment."
        ) from error
    return SimpleRunner


def find_torch_module(root: Any) -> Optional[nn.Module]:
    queue = [root]
    visited: set[int] = set()
    for _ in range(12):
        if not queue:
            break
        item = queue.pop(0)
        if item is None or id(item) in visited:
            continue
        visited.add(id(item))
        if isinstance(item, nn.Module):
            return item
        for attribute in ("model", "_model", "module", "_module"):
            try:
                child = getattr(item, attribute, None)
            except Exception:
                child = None
            if child is not None:
                queue.append(child)
    return None


def runner_metadata(runner: Any) -> Dict[str, Any]:
    module = find_torch_module(runner)
    if module is None:
        return {"parameter_count": None, "trainable_parameter_count": None}
    parameters = list(module.parameters())
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "module_class": f"{module.__class__.__module__}.{module.__class__.__name__}",
    }


def run_forecast(
    runner: Any,
    input_state: Mapping[str, Any],
    lead_time: int,
    device: torch.device,
) -> Tuple[List[Any], float, float]:
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    states = list(runner.run(input_state=input_state, lead_time=lead_time))
    synchronize(device)
    elapsed_s = time.perf_counter() - started
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else 0.0
    )
    if not states:
        raise RuntimeError("The runner returned no forecast state.")
    return states, elapsed_s, peak_mb


def state_value(state: Any, key: str) -> Any:
    if isinstance(state, Mapping):
        return state.get(key)
    return getattr(state, key, None)


def output_fields(state: Any) -> Mapping[str, Any]:
    fields = state_value(state, "fields")
    if not isinstance(fields, Mapping):
        raise TypeError(
            f"Forecast state does not expose a field mapping; got {type(fields)}."
        )
    return fields


def finite_and_field_stats(
    fields: Mapping[str, Any]
) -> Tuple[float, int, Dict[str, Dict[str, Any]]]:
    finite_count = 0
    value_count = 0
    grid_points: Optional[int] = None
    selected: Dict[str, Dict[str, Any]] = {}
    for name, values in fields.items():
        array = np.asarray(values)
        if array.ndim != 1:
            array = array.reshape(-1)
        if grid_points is None:
            grid_points = int(array.size)
        elif array.size != grid_points:
            raise ValueError(
                f"Output field {name} has {array.size} points; expected {grid_points}."
            )
        finite = np.isfinite(array)
        finite_count += int(finite.sum())
        value_count += int(array.size)
        if name in SELECTED_OUTPUT_FIELDS:
            finite_values = array[finite]
            selected[name] = {
                "shape": list(np.asarray(values).shape),
                "minimum": float(finite_values.min()) if finite_values.size else None,
                "maximum": float(finite_values.max()) if finite_values.size else None,
                "mean": float(finite_values.mean()) if finite_values.size else None,
                "standard_deviation": (
                    float(finite_values.std()) if finite_values.size else None
                ),
                "finite_fraction": float(finite.mean()) if finite.size else 0.0,
            }
    return (
        float(finite_count / value_count) if value_count else 0.0,
        int(grid_points or 0),
        selected,
    )


def forecast_change_metrics(
    input_fields: Mapping[str, np.ndarray], forecast_fields: Mapping[str, Any]
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for name in ("2t", "msl", "t_500", "z_500", "10u", "10v"):
        if name not in input_fields or name not in forecast_fields:
            continue
        current = np.asarray(input_fields[name][-1], dtype=np.float64)
        forecast = np.asarray(forecast_fields[name], dtype=np.float64).reshape(-1)
        if current.shape != forecast.shape:
            continue
        delta = forecast - current
        finite = np.isfinite(delta)
        if not finite.any():
            continue
        results[name] = {
            "mean_change": float(delta[finite].mean()),
            "mean_absolute_change": float(np.abs(delta[finite]).mean()),
            "root_mean_square_change": float(
                np.sqrt(np.square(delta[finite]).mean())
            ),
        }
    return results


def physical_range_checks(fields: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    definitions = {
        "2t": (180.0, 340.0, "K"),
        "msl": (80_000.0, 110_000.0, "Pa"),
        "q_500": (0.0, 0.05, "kg kg-1"),
        "tcc": (0.0, 1.0, "1"),
    }
    checks: Dict[str, Dict[str, Any]] = {}
    for name, (lower, upper, unit) in definitions.items():
        if name not in fields:
            continue
        values = np.asarray(fields[name], dtype=np.float64).reshape(-1)
        finite = np.isfinite(values)
        in_range = finite & (values >= lower) & (values <= upper)
        checks[name] = {
            "lower": lower,
            "upper": upper,
            "unit": unit,
            "fraction_in_range": float(in_range.sum() / values.size),
            "interpretation": (
                "Broad diagnostic range only; not a forecast-accuracy criterion."
            ),
        }
    return checks


def load_target_metrics(
    path: Path, forecast_fields: Mapping[str, Any]
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Target NPZ not found: {path}")
    per_field: Dict[str, Dict[str, float]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            key = name.removeprefix("field__")
            if key not in forecast_fields or key in {"date", "latitudes", "longitudes"}:
                continue
            target = np.asarray(archive[name], dtype=np.float64).reshape(-1)
            forecast = np.asarray(
                forecast_fields[key], dtype=np.float64
            ).reshape(-1)
            if target.shape != forecast.shape:
                continue
            valid = np.isfinite(target) & np.isfinite(forecast)
            if not valid.any():
                continue
            difference = forecast[valid] - target[valid]
            per_field[key] = {
                "valid_points": int(valid.sum()),
                "mae": float(np.abs(difference).mean()),
                "rmse": float(np.sqrt(np.square(difference).mean())),
                "bias": float(difference.mean()),
            }
    return {
        "target_path": str(path),
        "per_field_unweighted": per_field,
        "note": (
            "Direct unweighted field errors. Operational scientific verification "
            "requires variable-specific units, masks, area weighting and a defined analysis target."
        ),
    }


def save_output_sample(
    path: Path,
    state: Any,
    fields: Mapping[str, Any],
    sample_points: int = 8192,
) -> None:
    first = np.asarray(next(iter(fields.values()))).reshape(-1)
    indexes = np.linspace(
        0, first.size - 1, min(sample_points, first.size), dtype=np.int64
    )
    payload: Dict[str, Any] = {
        "date": np.asarray(str(state_value(state, "date")), dtype="U32"),
        "point_indexes": indexes,
    }
    latitudes = state_value(state, "latitudes")
    longitudes = state_value(state, "longitudes")
    if latitudes is not None:
        payload["latitudes"] = np.asarray(latitudes).reshape(-1)[indexes]
    if longitudes is not None:
        payload["longitudes"] = np.asarray(longitudes).reshape(-1)[indexes]
    for name in SELECTED_OUTPUT_FIELDS:
        if name in fields:
            payload[name] = np.asarray(fields[name]).reshape(-1)[indexes]
    np.savez_compressed(path, **payload)


def save_full_output(path: Path, state: Any, fields: Mapping[str, Any]) -> None:
    payload: Dict[str, Any] = {
        "date": np.asarray(str(state_value(state, "date")), dtype="U32")
    }
    for name, values in fields.items():
        payload[str(name)] = np.asarray(values)
    latitudes = state_value(state, "latitudes")
    longitudes = state_value(state, "longitudes")
    if latitudes is not None:
        payload["latitudes"] = np.asarray(latitudes)
    if longitudes is not None:
        payload["longitudes"] = np.asarray(longitudes)
    np.savez_compressed(path, **payload)


def display_field(name: str, values: np.ndarray) -> Tuple[np.ndarray, str]:
    values = np.asarray(values)
    if name in {"2t", "t_500"}:
        return values - 273.15, f"{name} (°C)"
    if name == "msl":
        return values / 100.0, "msl (hPa)"
    if name == "z_500":
        return values / 9.80665, "z_500 (m)"
    if name == "tp":
        return values * 1000.0, "tp (mm)"
    return values, name


def save_plot(path: Path, state: Any, fields: Mapping[str, Any]) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    names = [name for name in ("2t", "msl", "t_500", "tp") if name in fields]
    if not names:
        return None
    latitudes = state_value(state, "latitudes")
    longitudes = state_value(state, "longitudes")
    first = np.asarray(fields[names[0]]).reshape(-1)
    sample_size = min(45_000, first.size)
    indexes = np.linspace(0, first.size - 1, sample_size, dtype=np.int64)
    figure, axes = plt.subplots(
        1, len(names), figsize=(4.4 * len(names), 3.6), squeeze=False
    )
    for axis, name in zip(axes[0], names):
        displayed, title = display_field(name, np.asarray(fields[name]).reshape(-1))
        if latitudes is not None and longitudes is not None:
            lat = np.asarray(latitudes).reshape(-1)[indexes]
            lon = np.asarray(longitudes).reshape(-1)[indexes]
            lon = ((lon + 180.0) % 360.0) - 180.0
            artist = axis.scatter(
                lon,
                lat,
                c=displayed[indexes],
                s=0.32,
                linewidths=0,
                cmap="coolwarm" if name in {"2t", "t_500"} else "viridis",
                rasterized=True,
            )
            axis.set_xlim(-180, 180)
            axis.set_ylim(-90, 90)
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
        else:
            artist = axis.scatter(
                indexes,
                displayed[indexes],
                c=displayed[indexes],
                s=0.5,
                linewidths=0,
                cmap="viridis",
                rasterized=True,
            )
            axis.set_xlabel("N320 node index")
        axis.set_title(title)
        figure.colorbar(artist, ax=axis, shrink=0.82)
    figure.suptitle("AIFS Single v1.0 — final forecast step", y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return str(path.resolve())


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_summary(path: Path, results: Mapping[str, Any]) -> None:
    paired = results.get("paired_target_metrics")
    lines = [
        "ECMWF AIFS Single v1.0 validation",
        "=" * 40,
        f"Status: {results['status']}",
        f"Validation level: {results['validation_level']}",
        f"Data source: {results['data_source']}",
        f"Device: {results['environment']['device']}",
        f"Checkpoint verified: {results['checkpoint']['checksum_verified']}",
        f"Official runner load: {results['model']['official_runner_load']}",
        f"Input fields: {results['input']['field_count']}",
        f"Input shape per field: {tuple(results['input']['shape_per_field'])}",
        f"Forecast steps: {results['output']['forecast_steps']}",
        f"Output fields: {results['output']['field_count']}",
        f"Grid points: {results['output']['grid_points']}",
        f"Finite output: {results['output']['finite_fraction']:.6f}",
        f"Median forecast latency: {results['performance']['median_latency_seconds']:.3f} s",
        f"Peak GPU memory: {results['performance']['peak_gpu_memory_mb']:.2f} MB",
    ]
    parameter_count = results["model"].get("parameter_count")
    if parameter_count is not None:
        lines.append(f"Parameters: {parameter_count:,}")
    if paired and paired.get("per_field_unweighted"):
        lines.append(
            "Scientific accuracy: paired target supplied; see metrics.json for per-field MAE/RMSE."
        )
    else:
        qualifier = (
            "synthetic input only"
            if results["data_source"].startswith("synthetic")
            else "no paired verifying analysis"
        )
        lines.append(f"Scientific accuracy: unavailable ({qualifier})")
    lines.append(f"Results: {results['output_dir']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.lead_time <= 0 or args.lead_time % 6 != 0:
        raise ValueError("--lead-time must be a positive multiple of 6 hours.")
    if args.chunks <= 0:
        raise ValueError("--chunks must be positive.")

    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else model_dir / CHECKPOINT_NAME
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else model_dir / "validation_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    print(f"Python:  {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {device}")
    print("Verifying the official checkpoint before any pickle loading...")
    checkpoint_metadata = verify_checkpoint(checkpoint_path, args.skip_checksum)
    print(
        "Checkpoint SHA256: "
        + (
            "VERIFIED"
            if checkpoint_metadata["checksum_verified"]
            else "NOT VERIFIED (--skip-checksum)"
        )
    )
    pyg_compatibility = install_torch_geometric_checkpoint_compatibility()
    if pyg_compatibility["applied"]:
        print(
            "PyG compatibility:   installed legacy Inspector state adapter "
            "for torch-geometric 2.7"
        )
    elif pyg_compatibility["status"] == "unavailable":
        print(
            "PyG compatibility:   unavailable - "
            f"{pyg_compatibility.get('error', 'unknown error')}",
            file=sys.stderr,
        )

    if args.preflight_only:
        dependencies = dependency_audit()
        required_failures = [
            name
            for name, item in dependencies.items()
            if item["required_for_default_inference"] and not item["import_ok"]
        ]
        if pyg_compatibility["status"] == "unavailable":
            required_failures.append("torch_geometric Inspector compatibility")
        preflight_status = "PASS" if not required_failures else "ENVIRONMENT_INCOMPLETE"
        print("\n========== AIFS Single v1.0 preflight ==========")
        print(f"Status:              {preflight_status}")
        print(f"Checkpoint bytes:    {checkpoint_metadata['bytes']:,}")
        print(f"Checksum verified:   {checkpoint_metadata['checksum_verified']}")
        print(f"CUDA available:      {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device:         {torch.cuda.get_device_name(0)}")
        print("Dependency imports:")
        for name, item in dependencies.items():
            requirement = "required" if item["required_for_default_inference"] else "optional"
            state = "OK" if item["import_ok"] else "MISSING/FAILED"
            version = item["version"] or "unknown"
            print(f"  {name:<18} {state:<14} {version:<12} ({requirement})")
            if not item["import_ok"]:
                print(f"    {item['error']}")
        if required_failures:
            print(
                "Action: report these required packages to the administrator; "
                "do not create a new environment or reinstall PyTorch/CUDA:"
            )
            print("  " + ", ".join(required_failures))
        print("Inference:            not run (--preflight-only)")
        return 0 if not required_failures else 2

    if device.type != "cuda":
        print(
            "WARNING: the model-card environment uses CUDA and FlashAttention. "
            "CPU execution may fail or require a newer SDPA-compatible Anemoi setup.",
            file=sys.stderr,
        )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(args.chunks)

    input_path: Optional[Path] = (
        args.input.expanduser().resolve() if args.input else None
    )
    source = args.data_source
    if source == "auto":
        input_path = input_path or find_local_npz(model_dir, output_dir)
        source = "npz" if input_path else "synthetic"
    elif input_path and source == "synthetic":
        raise ValueError("--input cannot be combined with --data-source synthetic.")
    elif input_path and source == "open-data":
        raise ValueError("--input cannot be combined with --data-source open-data.")
    elif source == "npz" and input_path is None:
        input_path = find_local_npz(model_dir, output_dir)
        if input_path is None:
            raise FileNotFoundError(
                "--data-source npz requested, but no local NPZ input was found."
            )
    elif input_path is not None:
        source = "npz"

    if source == "synthetic":
        initial_date = parse_date(args.date, dt.datetime(2024, 1, 1, 0, 0))
        input_state, input_metadata = make_synthetic_state(args.seed, initial_date)
        data_source = "synthetic_n320_weather_like"
        validation_level = "functional_only"
        save_input_sample(
            output_dir / "synthetic_input_sample_for_audit.npz", input_state
        )
    elif source == "npz":
        assert input_path is not None
        input_state, input_metadata = load_npz_state(input_path, args.date)
        data_source = "local_npz_initial_state"
        validation_level = "paired_scientific" if args.target else "functional_only"
    else:
        input_state, input_metadata = fetch_open_data_state(args.date)
        data_source = "ecmwf_open_data"
        validation_level = "paired_scientific" if args.target else "functional_only"

    if args.save_full_input:
        save_state_npz(output_dir / "input_state_full.npz", input_state)

    SimpleRunner = import_simple_runner()
    print("Loading the checksum-verified checkpoint with the official Anemoi runner...")
    load_started = time.perf_counter()
    runner = SimpleRunner(str(checkpoint_path), device=str(device))
    synchronize(device)
    load_seconds = time.perf_counter() - load_started
    model_metadata = runner_metadata(runner)
    model_metadata.update(
        {
            "name": MODEL_NAME,
            "official_runner": "anemoi.inference.runners.simple.SimpleRunner",
            "official_runner_load": True,
            "load_seconds": load_seconds,
            "resolution": "N320 (~31 km), 542080 grid points",
            "forecast_step_hours": 6,
        }
    )

    for _ in range(max(args.warmup, 0)):
        run_forecast(runner, input_state, args.lead_time, device)

    timings: List[float] = []
    peaks: List[float] = []
    final_states: List[Any] = []
    for _ in range(max(args.runs, 1)):
        states, elapsed_s, peak_mb = run_forecast(
            runner, input_state, args.lead_time, device
        )
        timings.append(elapsed_s)
        peaks.append(peak_mb)
        final_states = states

    final_state = final_states[-1]
    fields = output_fields(final_state)
    finite_fraction, grid_points, selected_stats = finite_and_field_stats(fields)
    if grid_points != GRID_POINTS:
        raise ValueError(
            f"Forecast contains {grid_points} grid points; expected {GRID_POINTS}."
        )
    change_metrics = forecast_change_metrics(input_state["fields"], fields)
    range_checks = physical_range_checks(fields)
    paired_metrics = (
        load_target_metrics(args.target.expanduser().resolve(), fields)
        if args.target
        else None
    )

    sample_path = output_dir / "forecast_output_sample.npz"
    save_output_sample(sample_path, final_state, fields)
    full_output_path = None
    if args.save_full_output:
        full_output_path = output_dir / "forecast_output_full.npz"
        save_full_output(full_output_path, final_state, fields)
    preview = None
    if not args.no_plot:
        preview = save_plot(output_dir / "validation_preview.png", final_state, fields)

    forecast_steps = len(final_states)
    output_date = state_value(final_state, "date")
    status = (
        "PASS"
        if finite_fraction == 1.0
        and len(fields) > 0
        and forecast_steps >= 1
        and (checkpoint_metadata["checksum_verified"] or args.skip_checksum)
        else "FAIL"
    )
    results: Dict[str, Any] = {
        "status": status,
        "validation_level": validation_level,
        "model_name": MODEL_NAME,
        "data_source": data_source,
        "checkpoint": checkpoint_metadata,
        "model": model_metadata,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "anemoi_inference": package_version("anemoi-inference"),
            "anemoi_models": package_version("anemoi-models"),
            "flash_attn": package_version("flash-attn"),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "anemoi_inference_num_chunks": args.chunks,
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "torch_geometric_checkpoint_compatibility": pyg_compatibility,
        },
        "input": {
            **input_metadata,
            "field_names": list(INPUT_FIELDS),
            "field_count": len(INPUT_FIELDS),
            "shape_per_field": [2, GRID_POINTS],
            "input_path": str(input_path) if input_path else None,
            "full_input_saved": args.save_full_input,
        },
        "output": {
            "forecast_steps": forecast_steps,
            "lead_time_hours": args.lead_time,
            "final_date": str(output_date),
            "field_count": len(fields),
            "field_names": sorted(str(name) for name in fields),
            "grid_points": grid_points,
            "shape_per_field": [grid_points],
            "finite_fraction": finite_fraction,
            "selected_field_statistics": selected_stats,
            "forecast_change_from_initial_t0": change_metrics,
            "broad_physical_range_checks": range_checks,
            "sample_npz": str(sample_path.resolve()),
            "full_output_npz": (
                str(full_output_path.resolve()) if full_output_path else None
            ),
            "preview": preview,
        },
        "performance": {
            "checkpoint_load_seconds": load_seconds,
            "warmup_runs": max(args.warmup, 0),
            "timed_runs": max(args.runs, 1),
            "latencies_seconds": timings,
            "median_latency_seconds": float(statistics.median(timings)),
            "mean_latency_seconds": float(statistics.mean(timings)),
            "peak_gpu_memory_mb_per_run": peaks,
            "peak_gpu_memory_mb": float(max(peaks) if peaks else 0.0),
        },
        "paired_target_metrics": paired_metrics,
        "metric_interpretation": (
            "The synthetic input verifies checkpoint loading, model execution, "
            "output structure, numerical finiteness, latency and memory only. "
            "It does not measure AIFS scientific forecast skill."
            if data_source.startswith("synthetic")
            else (
                "A paired target was supplied; metrics.json includes direct "
                "unweighted field errors. Operational verification requires a "
                "defined analysis target and ECMWF-style weighting."
                if paired_metrics
                else "No paired verifying analysis was supplied, so scientific "
                "forecast accuracy is unavailable."
            )
        ),
        "official_reference": {
            "training_target": "6-hour atmospheric-state forecast",
            "verification_metrics": (
                "ACC, RMSE, SEEPS and forecast-activity diagnostics against "
                "operational analyses and observations"
            ),
            "note": (
                "The model card presents scorecards rather than a single scalar "
                "accuracy. Those published values are not recomputed here."
            ),
        },
        "output_dir": str(output_dir.resolve()),
    }
    results = json_ready(results)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(output_dir / "summary.txt", results)

    print("\n========== ECMWF AIFS Single v1.0 validation ==========")
    print(f"Status:              {status}")
    print(f"Validation level:    {validation_level}")
    print(f"Data source:         {data_source}")
    print(f"Device:              {device}")
    print(f"Checkpoint verified: {checkpoint_metadata['checksum_verified']}")
    print("Official runner load: True")
    print(f"Input fields:        {len(INPUT_FIELDS)}")
    print(f"Input field shape:   {(2, GRID_POINTS)}")
    print(f"Forecast steps:      {forecast_steps}")
    print(f"Output fields:       {len(fields)}")
    print(f"Output field shape:  {(GRID_POINTS,)}")
    parameter_count = model_metadata.get("parameter_count")
    if parameter_count is not None:
        print(f"Parameters:          {parameter_count:,}")
    print(f"Finite output:       {finite_fraction:.6f}")
    print(f"Median latency:      {statistics.median(timings):.3f} s")
    print(f"Peak GPU memory:     {max(peaks) if peaks else 0.0:.2f} MB")
    if selected_stats:
        compact = ", ".join(
            f"{name} mean={stats['mean']:.6g}"
            for name, stats in selected_stats.items()
            if stats.get("mean") is not None
        )
        print(f"Selected outputs:    {compact}")
    if paired_metrics and paired_metrics.get("per_field_unweighted"):
        print("Scientific accuracy: paired target metrics written to metrics.json")
    elif data_source.startswith("synthetic"):
        print("Scientific accuracy: unavailable (synthetic input only)")
    else:
        print("Scientific accuracy: unavailable (no paired verifying analysis)")
    print(f"Results:             {output_dir.resolve()}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
