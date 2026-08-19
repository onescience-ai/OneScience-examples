"""OneScience adapter for the official Microsoft Aurora implementation.

The neural network is the flattened Microsoft Aurora implementation in this directory. This
module owns only the contract between the flat ERA5 tensor used by OneScience and Aurora's
structured batch object.
"""

from __future__ import annotations

from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn


_TORCHVISION_SCHEMA_LIBRARY: torch.library.Library | None = None


def _operator_schema_exists(name: str) -> bool:
    try:
        torch._C._dispatch_has_kernel_for_dispatch_key(name, "Meta")
    except RuntimeError:
        return False
    return True


def _register_missing_torchvision_schemas() -> None:
    """Allow timm imports when the site torchvision extension cannot be loaded."""
    global _TORCHVISION_SCHEMA_LIBRARY
    spec = find_spec("torchvision")
    if spec is not None and spec.submodule_search_locations:
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        for extension in sorted(package_dir.glob("_C*.so")):
            try:
                torch.ops.load_library(str(extension))
            except (OSError, RuntimeError):
                continue
            break
    schemas = {
        "torchvision::nms": "nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor",
        "torchvision::qnms": "qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor",
    }
    missing = [schema for name, schema in schemas.items() if not _operator_schema_exists(name)]
    if not missing:
        return
    library = torch.library.Library("torchvision", "FRAGMENT")
    for schema in missing:
        library.define(schema)
    _TORCHVISION_SCHEMA_LIBRARY = library


_register_missing_torchvision_schemas()

# The implementation modules are imported here as a single public model surface.  Keeping the
# official classes in their original modules preserves their state-dict names and checkpoint
# compatibility while all callers use this file as the model entry point.
from .aurora_area import area, expand_matrix, radius_earth
from .aurora_batch import Batch, Metadata, interpolate, interpolate_numpy
from .aurora_compat import *
from .aurora_decoder import LinearPatchReconstruction, Perceiver3DDecoder
from .aurora_encoder import Perceiver3DEncoder
from .aurora_film import AdaptiveLayerNorm
from .aurora_fourier import *
from .aurora_insolation import insolation
from .aurora_levelcond import LevelConditioned
from .aurora_lora import *
from .aurora_network import (
    Aurora,
    Aurora12hPretrained,
    AuroraAirPollution,
    AuroraHighRes,
    AuroraPretrained,
    AuroraSmall,
    AuroraSmallPretrained,
    AuroraV1p5,
    AuroraV1p5Ensemble,
    AuroraWave,
)
from .aurora_normalisation import *
from .aurora_patchembed import LevelPatchEmbed
from .aurora_perceiver import MLP, PerceiverAttention, PerceiverResampler
from .aurora_posencoding import *
from .aurora_rollout import rollout as aurora_rollout

# Keep the official public API available from the unified model entry point.
rollout = aurora_rollout
from .aurora_swin3d import Swin3DTransformerBackbone, Swin3DTransformerBlock, WindowAttention
from .aurora_util import *


AURORA_SURFACE_VARS = ("2t", "10u", "10v", "msl")
AURORA_STATIC_VARS = ("lsm", "z", "slt")
AURORA_ATMOS_VARS = ("z", "u", "v", "t", "q")
AURORA_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

ERA5_SURFACE_MAPPING = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
ERA5_ATMOS_MAPPING = {
    "z": "geopotential_{level}",
    "u": "u_component_of_wind_{level}",
    "v": "v_component_of_wind_{level}",
    "t": "temperature_{level}",
    "q": "specific_humidity_{level}",
}


def _channel_order() -> tuple[str, ...]:
    channels = [ERA5_SURFACE_MAPPING[name] for name in AURORA_SURFACE_VARS]
    for name in AURORA_ATMOS_VARS:
        channels.extend(ERA5_ATMOS_MAPPING[name].format(level=level) for level in AURORA_LEVELS)
    return tuple(channels)


DEFAULT_ERA5_CHANNEL_ORDER = _channel_order()


def _as_float_tensor(value: torch.Tensor | np.ndarray, *, name: str) -> torch.Tensor:
    """Convert a field to float32 without changing its spatial dimensions."""
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim != 2:
        raise ValueError(f"Static field {name!r} must have shape [H, W], got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32)


def _parse_time(value: datetime | str) -> datetime:
    """Parse OneScience's ``YYYYmmddHH`` index and common ISO timestamps."""
    if isinstance(value, datetime):
        return value
    text = str(value)
    for fmt in ("%Y%m%d%H", "%Y%m%d%H%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported Aurora timestamp {value!r}")


def _normalise_times(times: datetime | str | Sequence[datetime | str], batch_size: int) -> tuple[datetime, ...]:
    if isinstance(times, (datetime, str)):
        parsed = (_parse_time(times),) * batch_size
    else:
        parsed = tuple(_parse_time(value) for value in times)
        if len(parsed) != batch_size:
            raise ValueError(f"Expected {batch_size} timestamps, received {len(parsed)}")
    return parsed


def _default_coordinates(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    lat = torch.linspace(90.0, -90.0, height, dtype=torch.float32)
    lon = torch.arange(width, dtype=torch.float32) * (360.0 / width)
    return lat, lon


class AuroraOneScience(nn.Module):
    """Aurora with a strict OneScience ERA5 tensor interface.

    Args:
        variant: ``"small"`` selects the official small architecture; ``"base"`` selects the
            official full 0.25-degree architecture. Both are randomly initialized unless a local
            checkpoint is explicitly supplied.
        static_path: Optional ``.npz`` containing ``lsm``, ``z``, ``slt``, ``lat`` and ``lon``.
        channel_order: Must equal the configured 69-channel ERA5 order exactly.
        load_checkpoint: If true, load ``checkpoint_path`` after constructing the network.
    """

    def __init__(
        self,
        *,
        variant: str = "small",
        static_path: str | Path | None = None,
        channel_order: Sequence[str] = DEFAULT_ERA5_CHANNEL_ORDER,
        atmos_levels: Sequence[int | float] = AURORA_LEVELS,
        checkpoint_path: str | Path | None = None,
        load_checkpoint: bool = False,
        checkpoint_strict: bool = True,
        network_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.channel_order = tuple(channel_order)
        if self.channel_order != DEFAULT_ERA5_CHANNEL_ORDER:
            raise ValueError(
                "Aurora requires the exact ERA5 channel order defined by "
                "DEFAULT_ERA5_CHANNEL_ORDER"
            )
        self.atmos_levels = tuple(atmos_levels)
        if self.atmos_levels != AURORA_LEVELS:
            raise ValueError(f"Aurora requires pressure levels {AURORA_LEVELS}, got {self.atmos_levels}")

        kwargs = dict(network_kwargs or {})
        kwargs.setdefault("max_history_size", 2)
        if variant == "small":
            self.network = AuroraSmallPretrained(**kwargs)
        elif variant in {"base", "full", "pretrained"}:
            self.network = AuroraPretrained(**kwargs)
        else:
            raise ValueError(f"Unknown Aurora variant {variant!r}; expected 'small' or 'base'")

        self.register_buffer("_static_lsm", torch.empty(0), persistent=False)
        self.register_buffer("_static_z", torch.empty(0), persistent=False)
        self.register_buffer("_static_slt", torch.empty(0), persistent=False)
        self.register_buffer("_lat", torch.empty(0), persistent=False)
        self.register_buffer("_lon", torch.empty(0), persistent=False)
        if static_path is not None:
            self.load_static_fields(static_path)
        if load_checkpoint:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when load_checkpoint=True")
            self.load_checkpoint_local(checkpoint_path, strict=checkpoint_strict)

    @property
    def aurora(self) -> nn.Module:
        """Expose the official network for activation-checkpointing and inspection."""
        return self.network

    def load_static_fields(self, path: str | Path) -> None:
        """Load and validate project static fields from a compressed NumPy archive."""
        static_path = Path(path).expanduser()
        with np.load(static_path) as data:
            required = set(AURORA_STATIC_VARS) | {"lat", "lon"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"{static_path}: missing static fields {missing}")
            fields = {name: _as_float_tensor(data[name], name=name) for name in AURORA_STATIC_VARS}
            lat = torch.as_tensor(data["lat"], dtype=torch.float32)
            lon = torch.as_tensor(data["lon"], dtype=torch.float32)
        self.set_static_fields(fields, lat=lat, lon=lon)

    def set_static_fields(
        self,
        static_vars: Mapping[str, torch.Tensor | np.ndarray],
        *,
        lat: torch.Tensor | np.ndarray | None = None,
        lon: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Set static fields and coordinates without making them checkpoint parameters."""
        missing = sorted(set(AURORA_STATIC_VARS) - set(static_vars))
        if missing:
            raise ValueError(f"Missing Aurora static variables: {missing}")
        fields = {name: _as_float_tensor(static_vars[name], name=name) for name in AURORA_STATIC_VARS}
        shape = fields["lsm"].shape
        if any(value.shape != shape for value in fields.values()):
            raise ValueError("Aurora static fields must have identical [H, W] shapes")
        if lat is None or lon is None:
            default_lat, default_lon = _default_coordinates(*shape)
            lat = default_lat if lat is None else lat
            lon = default_lon if lon is None else lon
        lat = lat if isinstance(lat, torch.Tensor) else torch.as_tensor(lat)
        lon = lon if isinstance(lon, torch.Tensor) else torch.as_tensor(lon)
        lat = lat.to(dtype=torch.float32)
        lon = lon.to(dtype=torch.float32)
        if lat.shape != (shape[0],) or lon.shape != (shape[1],):
            raise ValueError(f"Coordinates must have shapes {(shape[0],)} and {(shape[1],)}")
        if not torch.all(lat[1:] < lat[:-1]) or not torch.all(lon[1:] > lon[:-1]):
            raise ValueError("Aurora coordinates must be decreasing latitude and increasing longitude")
        self._static_lsm = fields["lsm"]
        self._static_z = fields["z"]
        self._static_slt = fields["slt"]
        self._lat = lat
        self._lon = lon

    def load_checkpoint_local(self, path: str | Path, *, strict: bool = True) -> None:
        """Load an official Aurora checkpoint after model construction."""
        self.network.load_checkpoint_local(str(Path(path).expanduser()), strict=strict)

    def configure_activation_checkpointing(self) -> None:
        """Delegate official activation checkpointing configuration."""
        self.network.configure_activation_checkpointing()

    def _static_for_shape(
        self,
        shape: tuple[int, int],
        static_vars: Mapping[str, torch.Tensor | np.ndarray] | None,
        lat: torch.Tensor | np.ndarray | None,
        lon: torch.Tensor | np.ndarray | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if static_vars is not None:
            missing = sorted(set(AURORA_STATIC_VARS) - set(static_vars))
            if missing:
                raise ValueError(f"Missing Aurora static variables: {missing}")
            fields = {name: _as_float_tensor(static_vars[name], name=name) for name in AURORA_STATIC_VARS}
            lat_value = lat if lat is not None else self._lat
            lon_value = lon if lon is not None else self._lon
        else:
            fields = {
                "lsm": self._static_lsm,
                "z": self._static_z,
                "slt": self._static_slt,
            }
            lat_value = lat if lat is not None else self._lat
            lon_value = lon if lon is not None else self._lon
        if any(value.numel() == 0 for value in fields.values()):
            raise ValueError("Static fields are not configured; pass static_vars or static_path")
        if any(tuple(value.shape) != shape for value in fields.values()):
            raise ValueError(f"Static fields must match input grid {shape}")
        lat_tensor = lat_value if isinstance(lat_value, torch.Tensor) else torch.as_tensor(lat_value)
        lon_tensor = lon_value if isinstance(lon_value, torch.Tensor) else torch.as_tensor(lon_value)
        if lat_tensor.numel() == 0 or lon_tensor.numel() == 0:
            lat_tensor, lon_tensor = _default_coordinates(*shape)
        return fields, lat_tensor.to(torch.float32), lon_tensor.to(torch.float32)

    def tensor_to_batch(
        self,
        inputs: torch.Tensor,
        *,
        times: datetime | str | Sequence[datetime | str],
        static_vars: Mapping[str, torch.Tensor | np.ndarray] | None = None,
        lat: torch.Tensor | np.ndarray | None = None,
        lon: torch.Tensor | np.ndarray | None = None,
    ) -> Batch:
        """Convert ``[B, T, 69, H, W]`` ERA5 data to an official Aurora Batch."""
        if inputs.ndim != 5:
            raise ValueError(f"Aurora input must have shape [B, T, C, H, W], got {tuple(inputs.shape)}")
        batch_size, history, channels, height, width = inputs.shape
        if channels != len(self.channel_order):
            raise ValueError(f"Expected {len(self.channel_order)} channels, got {channels}")
        if history < 1 or history > 2:
            raise ValueError(f"Aurora supports one or two history frames, got {history}")
        fields, lat_tensor, lon_tensor = self._static_for_shape(
            (height, width), static_vars, lat, lon
        )
        channel_indices = {name: self.channel_order.index(name) for name in self.channel_order}
        surface = {
            name: inputs[:, :, channel_indices[ERA5_SURFACE_MAPPING[name]]]
            for name in AURORA_SURFACE_VARS
        }
        atmos = {}
        for name in AURORA_ATMOS_VARS:
            indices = [
                channel_indices[ERA5_ATMOS_MAPPING[name].format(level=level)]
                for level in self.atmos_levels
            ]
            atmos[name] = torch.stack([inputs[:, :, index] for index in indices], dim=2)
        metadata = Metadata(
            lat=lat_tensor,
            lon=lon_tensor,
            time=_normalise_times(times, batch_size),
            atmos_levels=self.atmos_levels,
        )
        return Batch(surface, fields, atmos, metadata)

    def batch_to_tensor(self, prediction: Batch) -> torch.Tensor:
        """Flatten a one-step Aurora prediction back to ``[B, 69, H, W]``."""
        surface = [prediction.surf_vars[name][:, -1] for name in AURORA_SURFACE_VARS]
        atmos = [
            prediction.atmos_vars[name][:, -1, level_index]
            for name in AURORA_ATMOS_VARS
            for level_index in range(len(self.atmos_levels))
        ]
        return torch.stack(surface + atmos, dim=1)

    @staticmethod
    def crop_target(target: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
        """Apply the same one-row latitude crop used by the official Batch.crop."""
        if target.ndim not in {4, 5}:
            raise ValueError(f"Target must have [B,C,H,W] or [B,T,C,H,W], got {tuple(target.shape)}")
        height, width = target.shape[-2:]
        if width % patch_size != 0:
            raise ValueError("Target longitude size must be divisible by the Aurora patch size")
        if height % patch_size == 0:
            return target
        if height % patch_size == 1:
            return target[..., :-1, :]
        raise ValueError("Aurora permits at most one extra latitude row")

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        times: datetime | str | Sequence[datetime | str],
        static_vars: Mapping[str, torch.Tensor | np.ndarray] | None = None,
        lat: torch.Tensor | np.ndarray | None = None,
        lon: torch.Tensor | np.ndarray | None = None,
        return_batch: bool = False,
    ) -> torch.Tensor | Batch:
        batch = self.tensor_to_batch(inputs, times=times, static_vars=static_vars, lat=lat, lon=lon)
        prediction = self.network(batch)
        return prediction if return_batch else self.batch_to_tensor(prediction)

    def rollout(
        self,
        inputs: torch.Tensor,
        *,
        times: datetime | str | Sequence[datetime | str],
        steps: int,
        static_vars: Mapping[str, torch.Tensor | np.ndarray] | None = None,
        lat: torch.Tensor | np.ndarray | None = None,
        lon: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Run official autoregressive rollout and return ``[B, steps, 69, H, W]``."""
        if steps < 1:
            raise ValueError(f"Rollout steps must be positive, got {steps}")
        batch = self.tensor_to_batch(inputs, times=times, static_vars=static_vars, lat=lat, lon=lon)
        predictions = [
            self.batch_to_tensor(prediction)
            for prediction in aurora_rollout(self.network, batch, steps=steps)
        ]
        return torch.stack(predictions, dim=1)


def build_aurora_model(
    config: Mapping[str, object],
    *,
    project_root: str | Path | None = None,
    load_pretrained: bool = False,
) -> AuroraOneScience:
    """Build the adapter from the project's YAML configuration mapping."""
    model_cfg = config["model"]
    data_cfg = config["data"]
    root = Path(project_root or ".").expanduser().resolve()
    static_value = data_cfg.get("static_file")
    static_path = None
    if static_value:
        static_path = Path(str(static_value)).expanduser()
        if not static_path.is_absolute():
            static_path = root / static_path
    checkpoint_value = model_cfg.get("checkpoint", {}).get("local_small")
    checkpoint_path = None
    if checkpoint_value:
        checkpoint_path = Path(str(checkpoint_value)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
    return AuroraOneScience(
        variant=str(model_cfg.get("variant", "small")),
        static_path=static_path,
        channel_order=tuple(data_cfg.get("channel_order", DEFAULT_ERA5_CHANNEL_ORDER)),
        atmos_levels=tuple(model_cfg.get("atmos_levels", AURORA_LEVELS)),
        checkpoint_path=checkpoint_path,
        load_checkpoint=load_pretrained,
    )
