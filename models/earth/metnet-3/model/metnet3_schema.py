"""Input/output contract for the isolated fake MetNet-3 package."""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

import torch


@dataclass
class InputSchema:
    high_channels: int = 4
    low_channels: int = 3
    omo_channels: int = 2
    hrrr_channels: int = 8  # ASSUMPTION: proxy, not the paper's 617 channels.
    goes_channels: int = 4  # ASSUMPTION: proxy, not the paper's 16 channels.
    static_channels: int = 2  # coordinates/topography grouping; elevation is explicit below.
    high_frames: int = 3  # ADAPTATION: paper describes 11 MRMS frames.
    omo_frames: int = 3  # ADAPTATION: paper describes 9 OMO frames.
    ground_targets: int = 6
    precipitation_bins: int = 16  # ADAPTATION: paper uses 512 bins.
    ground_bins: int = 8  # ADAPTATION: paper uses 256/180 bins.

    @property
    def total_static_channels(self) -> int:
        return self.static_channels + 2  # elevation/topography + coordinates


def _require(batch: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"missing MetNet-3 input: {key}")
    return batch[key]


def validate_batch(batch: Mapping[str, torch.Tensor], schema: InputSchema) -> Tuple[int, int, int]:
    """Validate fake/adapter tensors and return (batch, height, width)."""
    high = _require(batch, "mrms_high")
    if high.ndim != 5:
        raise ValueError("mrms_high must be [B,T,C,H,W]")
    b, _, _, h, w = high.shape
    for key in ("mrms_low", "omo", "hrrr", "goes"):
        value = _require(batch, key)
        if value.shape[0] != b or value.ndim != 5:
            raise ValueError(f"{key} must be batched [B,T,C,H,W]")
    for key in ("elevation", "coordinates", "topography_embedding", "omo_input_mask", "current_time", "lead_time"):
        value = _require(batch, key)
        if key in ("current_time", "lead_time"):
            if value.shape != (b, 1):
                raise ValueError(f"{key} must have shape [B, 1]")
        elif value.shape[-2:] != (h, w):
            raise ValueError(f"{key} spatial shape mismatch")
    return b, h, w
