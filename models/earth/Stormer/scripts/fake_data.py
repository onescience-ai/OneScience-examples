"""
Generate synthetic ERA5 data and normalization files for Stormer pipeline.

Creates:
1. HDF5 data files (lightweight, chunked with zero fill)
2. Static/invariant files (geopotential, land_sea_mask, lat, lon, etc.)
3. Normalization constants matching official Stormer format:
   - normalize_mean.npz, normalize_std.npz (input normalization, 109+ vars)
   - normalize_diff_mean_{t}.npz, normalize_diff_std_{t}.npz (diff normalization)

Expected output structure:
    data/
    ├── data/
    │   ├── {year}.h5
    │   └── ...
    ├── static/
    │   ├── geopotential.nc, land_sea_mask.nc
    │   ├── land_mask.npy, soil_type.npy, topography.npy
    │   ├── lat.npy, lon.npy
    └── normalize/
        ├── normalize_mean.npz, normalize_std.npz
        └── normalize_diff_mean_{6,12,24}.npz, normalize_diff_std_{6,12,24}.npz
"""

import os
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import h5py
import numpy as np
import xarray as xr

from onescience.utils.YParams import YParams

# Stormer data dimensions:
# T=20 gives 120hrs (5 days) per year — enough for 72h lead time validation
DATASET_DIMS = {"T": 20, "H": 128, "W": 256, "time_step": 6}

# Full variable list from WeatherBench2 (109 vars) — used for normalization files
WB2_ALL_VARS = [
    # Surface
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    # Geopotential at pressure levels
    "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200",
    "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500",
    "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925",
    "geopotential_1000",
    # U wind
    "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150",
    "u_component_of_wind_200", "u_component_of_wind_250", "u_component_of_wind_300",
    "u_component_of_wind_400", "u_component_of_wind_500", "u_component_of_wind_600",
    "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925",
    "u_component_of_wind_1000",
    # V wind
    "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150",
    "v_component_of_wind_200", "v_component_of_wind_250", "v_component_of_wind_300",
    "v_component_of_wind_400", "v_component_of_wind_500", "v_component_of_wind_600",
    "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925",
    "v_component_of_wind_1000",
    # Temperature
    "temperature_50", "temperature_100", "temperature_150", "temperature_200",
    "temperature_250", "temperature_300", "temperature_400", "temperature_500",
    "temperature_600", "temperature_700", "temperature_850", "temperature_925",
    "temperature_1000",
    # Specific humidity
    "specific_humidity_50", "specific_humidity_100", "specific_humidity_150",
    "specific_humidity_200", "specific_humidity_250", "specific_humidity_300",
    "specific_humidity_400", "specific_humidity_500", "specific_humidity_600",
    "specific_humidity_700", "specific_humidity_850", "specific_humidity_925",
    "specific_humidity_1000",
    # Additional WB2 variables (for normalization compatibility)
    "angle_of_sub_gridscale_orography", "anisotropy_of_sub_gridscale_orography",
    "geopotential_at_surface", "high_vegetation_cover", "lake_cover", "lake_depth",
    "land_sea_mask", "low_vegetation_cover", "orography",
    "slope_of_sub_gridscale_orography", "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography", "type_of_high_vegetation",
    "type_of_low_vegetation",
    "mean_surface_latent_heat_flux", "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux", "mean_surface_sensible_heat_flux",
    "mean_top_downward_short_wave_radiation_flux",
    "mean_top_net_long_wave_radiation_flux", "mean_top_net_short_wave_radiation_flux",
    "skin_temperature", "snow_depth", "10m_wind_speed", "surface_pressure",
    "toa_incident_solar_radiation", "total_precipitation_6hr",
    "total_column_water_vapour", "total_cloud_cover", "sea_ice_cover",
    "sea_surface_temperature", "vertical_velocity_50", "vertical_velocity_100",
    "vertical_velocity_150", "vertical_velocity_200", "vertical_velocity_250",
    "vertical_velocity_300", "vertical_velocity_400", "vertical_velocity_500",
    "vertical_velocity_600", "vertical_velocity_700", "vertical_velocity_850",
    "vertical_velocity_925", "vertical_velocity_1000",
]


def generate_fake_h5(data_dir, var_names, years, dims):
    """Generate HDF5 files with random values for pipeline validation.

    Writes actual random data (N(0, 1) per variable) to ensure non-zero
    training loss for verifying gradient flow.
    """
    os.makedirs(os.path.join(data_dir, "data"), exist_ok=True)
    T, C = dims["T"], len(var_names)
    H, W = dims["H"], dims["W"]

    for year in years:
        path = os.path.join(data_dir, "data", f"{year}.h5")

        # Generate random data with per-variable mean=0, std=1
        rng = np.random.default_rng(42 + year)
        fields = rng.normal(0, 1, (T, C, H, W)).astype(np.float32)

        # Add temporal correlation so consecutive frames are similar
        # (simple AR(1): smooth the time dimension)
        for t in range(1, T):
            fields[t] = 0.7 * fields[t-1] + 0.3 * fields[t]

        # Per-variable means/stds
        means = fields.mean(axis=(0, 2, 3), keepdims=True).reshape(1, C, 1, 1)
        stds = fields.std(axis=(0, 2, 3), keepdims=True).reshape(1, C, 1, 1)
        stds = np.maximum(stds, 0.01)  # avoid division by zero

        with h5py.File(path, "w") as f:
            ds = f.create_dataset(
                "fields",
                data=fields,
                dtype="float32",
                chunks=(1, C, H, W),
            )
            ds.attrs["variables"] = var_names
            ds.attrs["time_step"] = dims["time_step"]
            f.create_dataset("global_means", data=means)
            f.create_dataset("global_stds", data=stds)

        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {year}.h5  shape=({T},{C},{H},{W})  "
              f"size={size_mb:.1f}MB  logical={T*C*H*W*4/1024**3:.1f}GB")


def get_static(data_dir):
    """Generate synthetic static/invariant data files."""
    os.makedirs(data_dir, exist_ok=True)

    H, W = DATASET_DIMS["H"], DATASET_DIMS["W"]

    # Geopotential at surface
    ds = xr.Dataset(
        data_vars={
            "z": (("valid_time", "latitude", "longitude"),
                  np.random.rand(1, H, W).astype(np.float32))
        },
        coords={
            "valid_time": ["2015-12-31"],
            "latitude": np.linspace(90, -90, H, dtype=np.float64),
            "longitude": np.linspace(0, 359.75, W, dtype=np.float64),
            "number": 0, "expver": "",
        },
        attrs={
            "GRIB_centre": "ecmf",
            "Conventions": "CF-1.7",
            "institution": "European Centre for Medium-Range Weather Forecasts",
            "history": "Generated for Stormer onescience",
        }
    )
    ds.to_netcdf(f"{data_dir}/geopotential.nc")

    # Land-sea mask
    ds_lsm = xr.Dataset(
        data_vars={
            "lsm": (("valid_time", "latitude", "longitude"),
                    np.random.rand(1, H, W).astype(np.float32))
        },
        coords={
            "valid_time": ["2015-12-31"],
            "latitude": np.linspace(90, -90, H, dtype=np.float64),
            "longitude": np.linspace(0, 359.75, W, dtype=np.float64),
            "number": 0, "expver": "",
        },
    )
    ds_lsm.to_netcdf(f"{data_dir}/land_sea_mask.nc")

    # Static numpy arrays
    arr = np.random.randn(H, W).astype(np.float32)
    np.save(f'{data_dir}/land_mask.npy', arr)
    np.save(f'{data_dir}/soil_type.npy', arr)
    np.save(f'{data_dir}/topography.npy', arr)

    # Latitude/longitude arrays
    lat = np.linspace(90, -90, H, dtype=np.float32)
    np.save(f'{data_dir}/lat.npy', lat)
    lon = np.linspace(0, 359.75, W, dtype=np.float32)
    np.save(f'{data_dir}/lon.npy', lon)

    print(f"✅ Static data generated in {data_dir}")


def generate_normalization_files(normalize_dir, all_vars, stormer_vars):
    """Generate normalization .npz files matching official Stormer format.

    Creates files for ALL 109+ WB2 variables (for compatibility with official
    normalization loading code), plus diff normalization for intervals [6, 12, 24].

    For fake data: input mean=0, std=1; diff mean=0, diff_std scales with interval.
    """
    os.makedirs(normalize_dir, exist_ok=True)

    # ---- Input normalization ----
    # mean = 0, std = 1 for all variables (fake zero-mean unit-variance data)
    inp_mean = {v: np.array([0.0], dtype=np.float32) for v in all_vars}
    inp_std = {v: np.array([1.0], dtype=np.float32) for v in all_vars}

    np.savez(os.path.join(normalize_dir, "normalize_mean.npz"), **inp_mean)
    np.savez(os.path.join(normalize_dir, "normalize_std.npz"), **inp_std)
    print(f"  normalize_mean.npz: {len(inp_mean)} vars")
    print(f"  normalize_std.npz: {len(inp_std)} vars")

    # ---- Diff normalization for each interval ----
    # diff_std scales with sqrt(interval) (diffusion-like behavior in atmosphere)
    for interval in [6, 12, 24]:
        scale = np.sqrt(interval / 6.0)  # 6h→1.0, 12h→1.414, 24h→2.0
        diff_mean = {v: np.array([0.0], dtype=np.float32) for v in all_vars}
        diff_std = {v: np.array([scale], dtype=np.float32) for v in all_vars}

        np.savez(os.path.join(normalize_dir, f"normalize_diff_mean_{interval}.npz"),
                 **diff_mean)
        np.savez(os.path.join(normalize_dir, f"normalize_diff_std_{interval}.npz"),
                 **diff_std)
        print(f"  normalize_diff_*_{interval}.npz: {len(diff_std)} vars (scale={scale:.3f})")

    print(f"✅ Normalization files generated in {normalize_dir}")


if __name__ == "__main__":
    cfg_datapipe = YParams(os.path.join(str(root_path), "conf/config.yaml"), "datapipe")
    cfg_model = YParams(os.path.join(str(root_path), "conf/config.yaml"), "model")

    data_dir = cfg_datapipe.dataset.data_dir

    # Safety: refuse to overwrite real data paths
    if data_dir.startswith("/public/") or data_dir.startswith("/work2/") or data_dir.startswith("/work/"):
        print("❌ 请检查 config，确保 data_dir 指向本地测试路径而非生产路径。")
        sys.exit(1)

    years = (cfg_datapipe.dataset.train_time +
             cfg_datapipe.dataset.val_time +
             cfg_datapipe.dataset.test_time)
    stormer_vars = cfg_datapipe.dataset.channels

    # 1. Generate fake HDF5 data
    print(f"\n📂 Generating fake HDF5: {len(years)} years, {len(stormer_vars)} variables...")
    generate_fake_h5(data_dir, stormer_vars, years, DATASET_DIMS)

    # 2. Generate static files
    static_dir = os.path.join(data_dir, "static")
    print(f"\n📂 Generating static files...")
    get_static(static_dir)

    # 3. Generate normalization files
    normalize_dir = cfg_model.normalize_dir
    print(f"\n📂 Generating normalization files...")
    generate_normalization_files(normalize_dir, WB2_ALL_VARS, stormer_vars)

    print(f"\n✅ All fake datasets generated successfully.")
    print(f"   Data dir:    {data_dir}")
    print(f"   Norm dir:    {normalize_dir}")
    print(f"   Variables:   {len(stormer_vars)} (Stormer) / {len(WB2_ALL_VARS)} (WB2 full)")
    print(f"   Years:       {years}")
    print(f"   Resolution:  {DATASET_DIMS['H']}×{DATASET_DIMS['W']} (1.40625°)")
