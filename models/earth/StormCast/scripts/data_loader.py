from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import torch
from onescience.datapipes.climate.era5 import ERA5Dataset
from torch.utils.data import Dataset

from grid import lambert_grid


class StormCastDataset(Dataset):
    """Pair OneScience ERA5 backgrounds with synchronized local state targets."""

    def __init__(
        self,
        data_root: str | Path,
        years: list[int],
        era5_variables: list[str],
        state_variables: list[str],
        invariant_variables: list[str],
        image_size: list[int] | tuple[int, int],
        input_steps: int = 1,
        output_steps: int = 1,
        normalize: bool = True,
    ) -> None:
        if input_steps != 1 or output_steps != 1:
            raise ValueError("StormCast pairing currently requires one input and one target step")

        self.data_root = Path(data_root)
        self.years = years
        self.era5_variables = era5_variables
        self.state_variables = state_variables
        self.invariant_variables = invariant_variables
        self.image_size = tuple(image_size)
        self.normalize = normalize
        self.era5 = ERA5Dataset(
            dataset_dir=str(self.data_root / "era5"),
            used_years=years,
            used_variables=era5_variables,
            input_steps=input_steps,
            output_steps=output_steps,
            normalize=normalize,
        )
        self.samples_per_year = self.era5.samples_per_year
        self._validate_era5_grid()
        self._validate_local_files()
        self.invariants = self._load_invariants()
        self._initialize_background_regrid()

    def _validate_era5_grid(self) -> None:
        if self.era5.H < 2 or self.era5.W < 2:
            raise ValueError("ERA5 grid must have at least two points per dimension")
        expected = (721, 1440)
        if (self.era5.H, self.era5.W) != expected:
            raise ValueError(
                f"StormCast expects ERA5 on the global {expected} grid, "
                f"got {(self.era5.H, self.era5.W)}"
            )

    def _validate_local_files(self) -> None:
        for year in self.years:
            path = self.data_root / "hrrr" / "data" / f"{year}.h5"
            if not path.is_file():
                raise FileNotFoundError(f"Missing local state file: {path}")
            with h5py.File(path, "r") as handle:
                fields = handle["fields"]
                variables = [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in fields.attrs["variables"]
                ]
                if variables != self.state_variables:
                    raise ValueError(
                        "Local state channel order differs from data.state_variables"
                    )
                expected_steps = self.samples_per_year + 1
                if fields.shape[0] != expected_steps:
                    raise ValueError(
                        f"{path} has {fields.shape[0]} steps, expected {expected_steps}"
                    )
                if tuple(fields.shape[-2:]) != self.image_size:
                    raise ValueError(
                        f"Local state grid is {tuple(fields.shape[-2:])}, "
                        f"expected regional grid {self.image_size}"
                    )

    def _load_invariants(self) -> torch.Tensor:
        path = self.data_root / "hrrr" / "invariants.h5"
        with h5py.File(path, "r") as handle:
            fields = handle["fields"]
            variables = [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in fields.attrs["variables"]
            ]
            if variables != self.invariant_variables:
                raise ValueError(
                    "Invariant channel order differs from data.invariant_variables"
                )
            invariants = torch.as_tensor(fields[:], dtype=torch.float32)
            if tuple(invariants.shape[-2:]) != self.image_size:
                raise ValueError(
                    f"Invariant grid is {tuple(invariants.shape[-2:])}, "
                    f"expected {self.image_size}"
                )
            return invariants

    def _initialize_background_regrid(self) -> None:
        with h5py.File(self.data_root / "hrrr" / "invariants.h5", "r") as handle:
            if "lat" in handle and "lon" in handle:
                target_lat = torch.as_tensor(handle["lat"][:], dtype=torch.float32)
                target_lon = torch.as_tensor(handle["lon"][:], dtype=torch.float32)
            else:
                target_lat_np, target_lon_np = lambert_grid(self.image_size)
                target_lat = torch.from_numpy(target_lat_np)
                target_lon = torch.from_numpy(target_lon_np)
        if target_lat.shape != self.image_size or target_lon.shape != self.image_size:
            raise ValueError("StormCast target latitude/longitude grid has wrong shape")

        lat_position = (90.0 - target_lat) / (180.0 / (self.era5.H - 1))
        lon_position = torch.remainder(target_lon, 360.0) / (360.0 / self.era5.W)
        self.lat0 = lat_position.floor().long().clamp(0, self.era5.H - 2)
        self.lat1 = self.lat0 + 1
        self.lon0 = lon_position.floor().long().remainder(self.era5.W)
        self.lon1 = (self.lon0 + 1).remainder(self.era5.W)
        self.lat_weight = lat_position - self.lat0
        self.lon_weight = lon_position - lon_position.floor()

    def _regrid_background(self, background: torch.Tensor) -> torch.Tensor:
        f00 = background[..., self.lat0, self.lon0]
        f01 = background[..., self.lat0, self.lon1]
        f10 = background[..., self.lat1, self.lon0]
        f11 = background[..., self.lat1, self.lon1]
        lon_weight = self.lon_weight.to(background.dtype)
        lat_weight = self.lat_weight.to(background.dtype)
        top = torch.lerp(f00, f01, lon_weight)
        bottom = torch.lerp(f10, f11, lon_weight)
        return torch.lerp(top, bottom, lat_weight)

    def __len__(self) -> int:
        return len(self.era5)

    def __getitem__(self, index: int) -> dict[str, Any]:
        background, _, _, step_index, time_index = self.era5[index]
        background = self._regrid_background(background)
        year_index = index // self.samples_per_year
        year = self.years[year_index]
        path = self.data_root / "hrrr" / "data" / f"{year}.h5"

        with h5py.File(path, "r") as handle:
            state = torch.as_tensor(
                handle["fields"][step_index : step_index + 2], dtype=torch.float32
            )
            if self.normalize:
                means = torch.as_tensor(handle["global_means"][:], dtype=torch.float32)
                stds = torch.as_tensor(handle["global_stds"][:], dtype=torch.float32)
                state = (state - means) / stds

        return {
            "background": background,
            "state": (state[0], state[1]),
            "invariant": self.invariants,
            "step_index": step_index,
            "time_index": time_index,
        }
