"""OneScience ERA5 adapter for the official OneForecast 69-channel contract."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

SOURCE_GRID = (721, 1440)
ONEFORECAST_FILE_GRID = (121, 240)
SPATIAL_STRIDE = 6

OFFICIAL_VARIABLES = tuple(
    [f"Z{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
    + [f"Q{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
    + [f"T{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
    + [f"U{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
    + [f"V{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)]
    + ["U10M", "V10M", "T2M", "MSLP"]
)

VARIABLE_ALIASES = {
    **{f"Z{x}": f"geopotential_{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)},
    **{f"Q{x}": f"specific_humidity_{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)},
    **{f"T{x}": f"temperature_{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)},
    **{f"U{x}": f"u_component_of_wind_{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)},
    **{f"V{x}": f"v_component_of_wind_{x}" for x in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)},
    "U10M": "10m_u_component_of_wind",
    "V10M": "10m_v_component_of_wind",
    "T2M": "2m_temperature",
    "MSLP": "mean_sea_level_pressure",
}


def _decode_variables(values: Iterable[Any]) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


class OneForecastERA5Adapter:
    """Validate files and construct OneScience's ERA5 DataLoader."""

    def __init__(self, dataset_dir: str | Path, years: Iterable[int], batch_size: int = 1,
                 input_steps: int = 1, output_steps: int = 1, normalize: bool = True,
                 num_workers: int = 0, distributed: bool = False) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.years = [int(year) for year in years]
        self.batch_size = batch_size
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.normalize = normalize
        self.num_workers = num_workers
        self.distributed = distributed
        self.source_variables: list[str] = []
        self.channel_indices: list[int] = []
        self.global_means: np.ndarray | None = None
        self.global_stds: np.ndarray | None = None
        self.time_step_hours: int | None = None
        self.source_grid: tuple[int, int] | None = None
        self._external_stats: tuple[Path, Path] | None = None
        self._layout_dir: tempfile.TemporaryDirectory[str] | None = None
        self._validate_files()

    def _year_path(self, year: int) -> Path:
        for path in (self.dataset_dir / "data" / f"{year}.h5", self.dataset_dir / f"{year}.h5"):
            if path.is_file():
                return path
        raise FileNotFoundError(f"ERA5 file for year {year} was not found below {self.dataset_dir}")

    def _validate_files(self) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required to validate ERA5 HDF5 files") from exc
        if not self.years:
            raise ValueError("At least one ERA5 year is required")
        reference_variables: list[str] | None = None
        reference_indices: list[int] | None = None
        for year in self.years:
            path = self._year_path(year)
            with h5py.File(path, "r") as handle:
                if "fields" not in handle:
                    raise ValueError(f"{path} does not contain a fields dataset")
                fields = handle["fields"]
                if len(fields.shape) != 4:
                    raise ValueError(f"{path}: fields must have shape [T, C, H, W], got {fields.shape}")
                variables = _decode_variables(fields.attrs.get("variables", []))
                source_variables = [
                    name if name in variables else VARIABLE_ALIASES[name]
                    for name in OFFICIAL_VARIABLES
                    if name in variables or VARIABLE_ALIASES[name] in variables
                ]
                missing = [
                    name for name in OFFICIAL_VARIABLES
                    if name not in variables and VARIABLE_ALIASES[name] not in variables
                ]
                if missing:
                    raise ValueError(f"{path}: missing official variables: {missing}")
                indices = [variables.index(name) for name in source_variables]
                if reference_variables is not None and variables != reference_variables:
                    raise ValueError(f"{path}: complete variable metadata differs between yearly files")
                if reference_indices is not None and indices != reference_indices:
                    raise ValueError(f"{path}: official channel indices differ between yearly files")
                reference_variables, reference_indices = variables, indices
                self.source_variables = source_variables
                self.channel_indices = indices
                if fields.shape[1] != len(variables):
                    raise ValueError(f"{path}: variables metadata does not match channel dimension")
                if fields.shape[1] != 69 or tuple(fields.shape[2:]) not in (SOURCE_GRID, ONEFORECAST_FILE_GRID):
                    raise ValueError(
                        f"{path}: expected fields [T, 69, 721, 1440] or [T, 69, 121, 240], got {fields.shape}"
                    )
                grid = tuple(fields.shape[2:])
                if self.source_grid is not None and grid != self.source_grid:
                    raise ValueError(f"{path}: spatial grid differs between yearly files")
                self.source_grid = grid
                if fields.shape[0] < self.input_steps + self.output_steps:
                    raise ValueError(f"{path}: not enough time steps for configured window")
                if "time_step" not in fields.attrs:
                    raise ValueError(f"{path}: fields.attrs['time_step'] is required by ERA5Datapipe")
                time_step = int(fields.attrs["time_step"])
                if time_step != 6 or (self.time_step_hours is not None and time_step != self.time_step_hours):
                    raise ValueError(f"{path}: expected a consistent 6-hour time_step, got {time_step}")
                self.time_step_hours = time_step
                if "global_means" in handle and "global_stds" in handle:
                    means = np.asarray(handle["global_means"])
                    stds = np.asarray(handle["global_stds"])
                else:
                    candidates = (
                        (self.dataset_dir / "stats" / "global_means.npy",
                         self.dataset_dir / "stats" / "global_stds.npy"),
                        (self.dataset_dir / "mean.npy", self.dataset_dir / "std.npy"),
                        (self.dataset_dir.parent / "mean.npy", self.dataset_dir.parent / "std.npy"),
                    )
                    stats_paths = next(((mean, std) for mean, std in candidates
                                        if mean.is_file() and std.is_file()), None)
                    if stats_paths is None:
                        raise ValueError(f"{path}: embedded or external ERA5 statistics are required")
                    self._external_stats = stats_paths
                    means, stds = (np.load(item) for item in stats_paths)
                expected_shape = (1, len(variables), 1, 1)
                if means.shape != expected_shape or stds.shape != expected_shape:
                    raise ValueError(f"{path}: statistics must have shape {expected_shape}")
                if not np.isfinite(means).all() or not np.isfinite(stds).all() or not (stds > 0).all():
                    raise ValueError(f"{path}: statistics must be finite and standard deviations positive")
                if self.global_means is not None and not np.array_equal(means, self.global_means):
                    raise ValueError(f"{path}: global_means differ between yearly files")
                if self.global_stds is not None and not np.array_equal(stds, self.global_stds):
                    raise ValueError(f"{path}: global_stds differ between yearly files")
                self.global_means, self.global_stds = means, stds

    def _onescience_dataset_dir(self) -> Path:
        if self._layout_dir is not None:
            return Path(self._layout_dir.name)
        self._layout_dir = tempfile.TemporaryDirectory(prefix="oneforecast_era5_")
        root = Path(self._layout_dir.name)
        data_dir = root / "data"
        data_dir.mkdir()
        for year in self.years:
            source_path = self._year_path(year)
            target_path = data_dir / f"{year}.h5"
            if self.source_grid == SOURCE_GRID:
                import h5py

                with h5py.File(source_path, "r") as source_handle:
                    source_fields = source_handle["fields"]
                    layout = h5py.VirtualLayout(
                        shape=(source_fields.shape[0], source_fields.shape[1], *ONEFORECAST_FILE_GRID),
                        dtype=source_fields.dtype,
                    )
                    virtual_source = h5py.VirtualSource(str(source_path), "fields", shape=source_fields.shape)
                    layout[:] = virtual_source[:, :, ::SPATIAL_STRIDE, ::SPATIAL_STRIDE]
                    with h5py.File(target_path, "w", libver="latest") as target_handle:
                        fields = target_handle.create_virtual_dataset("fields", layout)
                        for name, value in source_fields.attrs.items():
                            fields.attrs[name] = value
            else:
                target_path.symlink_to(source_path)
        if self._external_stats is not None:
            stats_dir = root / "stats"
            stats_dir.mkdir()
            (stats_dir / "global_means.npy").symlink_to(self._external_stats[0])
            (stats_dir / "global_stds.npy").symlink_to(self._external_stats[1])

        return root

    def get_dataloader(self, mode: str):
        """Delegate loading to OneScience, then align native ERA5 to OneForecast's grid."""
        try:
            from onescience.datapipes.climate.era5 import ERA5Datapipe
        except ImportError as exc:
            raise RuntimeError("OneScience ERA5Datapipe is required for data loading") from exc
        datapipe = ERA5Datapipe(
            dataset_dir=str(self._onescience_dataset_dir()), used_years=self.years,
            used_variables=self.source_variables, distributed=self.distributed,
            input_steps=self.input_steps, output_steps=self.output_steps,
            normalize=self.normalize, batch_size=self.batch_size, num_workers=self.num_workers,
        )
        loader, sampler = datapipe.get_dataloader(mode=mode)
        return _SpatiallyAdaptedLoader(loader, self.source_grid), sampler

    def inspect(self) -> dict[str, Any]:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required to inspect ERA5 HDF5 files") from exc
        path = self._year_path(self.years[0])
        with h5py.File(path, "r") as handle:
            fields = handle["fields"]
            variables = _decode_variables(fields.attrs["variables"])
            indices = [variables.index(name) for name in self.source_variables]
            return {"path": str(path), "fields_shape": list(fields.shape),
                     "source_grid": list(fields.shape[2:]),
                     "oneforecast_file_grid": list(ONEFORECAST_FILE_GRID),
                     "oneforecast_model_grid": [120, 240],
                     "spatial_transform": "identity" if tuple(fields.shape[2:]) == ONEFORECAST_FILE_GRID else "stride_6",
                     "time_step_hours": int(fields.attrs["time_step"]),
                     "variable_count": len(variables), "official_channel_indices": indices,
                     "source_variables": self.source_variables,
                     "statistics_shape": list(self.global_means.shape),
                     "statistics_shared_across_years": True,
                     "official_variables_match": len(indices) == len(OFFICIAL_VARIABLES)}

    def selected_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """Return normalization statistics in the model's 69-channel order."""
        if self.global_means is None or self.global_stds is None:
            raise RuntimeError("ERA5 statistics have not been validated")
        return self.global_means[:, self.channel_indices], self.global_stds[:, self.channel_indices]


def _adapt_spatial(value: Any, source_grid: tuple[int, int] | None) -> Any:
    if not hasattr(value, "shape") or len(value.shape) < 2:
        return value
    if tuple(value.shape[-2:]) == ONEFORECAST_FILE_GRID:
        return value
    if tuple(value.shape[-2:]) != SOURCE_GRID or source_grid != SOURCE_GRID:
        return value
    return value[..., ::SPATIAL_STRIDE, ::SPATIAL_STRIDE]


class _SpatiallyAdaptedLoader:
    """Preserve the DataLoader interface while adapting fields after ERA5Datapipe."""

    def __init__(self, loader: Any, source_grid: tuple[int, int] | None) -> None:
        self.loader = loader
        self.source_grid = source_grid

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield tuple(_adapt_spatial(value, self.source_grid) for value in batch)
