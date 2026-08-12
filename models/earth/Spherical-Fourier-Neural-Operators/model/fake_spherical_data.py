"""Deterministic smooth spherical fields for structural SFNO tests."""

from __future__ import annotations

import math

import torch


def make_fake_spherical_sequence(
    timesteps: int,
    channels: int,
    nlat: int,
    nlon: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Create moving low-order spherical modes, not spatial white noise."""
    if min(timesteps, channels, nlat, nlon) <= 0:
        raise ValueError("All fake-data dimensions must be positive")

    generator = torch.Generator().manual_seed(seed)
    lat = torch.linspace(-math.pi / 2, math.pi / 2, nlat, dtype=dtype)
    lon = torch.arange(nlon, dtype=dtype) * (2 * math.pi / nlon)
    latitude, longitude = torch.meshgrid(lat, lon, indexing="ij")
    amplitudes = 0.8 + 0.4 * torch.rand(channels, generator=generator, dtype=dtype)
    phases = 2 * math.pi * torch.rand(channels, generator=generator, dtype=dtype)

    frames = []
    for step in range(timesteps):
        channel_fields = []
        for channel in range(channels):
            phase = phases[channel] + 0.18 * (channel + 1) * step
            zonal = torch.cos(latitude) * torch.cos((channel + 1) * longitude - phase)
            planetary = 0.35 * torch.sin(2 * latitude) * torch.sin(longitude + 0.11 * step)
            polar = 0.15 * torch.cos(3 * latitude - 0.07 * step)
            channel_fields.append(amplitudes[channel] * (zonal + planetary + polar))
        frames.append(torch.stack(channel_fields))

    fields = torch.stack(frames)
    means = fields.mean(dim=(0, 2, 3))
    stds = fields.std(dim=(0, 2, 3)).clamp_min(torch.finfo(dtype).eps)
    normalized = (fields - means[None, :, None, None]) / stds[None, :, None, None]
    return {
        "fields": normalized,
        "time": torch.arange(timesteps, dtype=dtype),
        "lat": torch.rad2deg(lat),
        "lon": torch.rad2deg(lon),
        "global_means": means,
        "global_stds": stds,
    }
