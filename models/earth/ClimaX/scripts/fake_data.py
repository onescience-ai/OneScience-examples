import os
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
import h5py
import numpy as np
from onescience.utils.YParams import YParams


# ClimaX uses 5.625° resolution: 32x64 grid, 6-hourly data
DATASET_DIMS = {"T": 10, "H": 32, "W": 64, "time_step": 6}


def generate_fake_h5(data_dir, var_names, years, dims):
    """
    Generate fake HDF5 files for each year matching the ERA5 format
    expected by onescience's ERA5Dataset.

    Uses HDF5 chunked datasets with fillvalue=0.0 — unallocated chunks
    return zeros, so files are tiny but have correct shapes.
    Mean/std are embedded in each year's h5 file.
    """
    os.makedirs(os.path.join(data_dir, "data"), exist_ok=True)
    T, C = dims["T"], len(var_names)
    H, W = dims["H"], dims["W"]

    means = np.zeros((1, C, 1, 1), dtype=np.float32)
    stds = np.ones((1, C, 1, 1), dtype=np.float32)

    for year in years:
        path = os.path.join(data_dir, "data", f"{year}.h5")
        with h5py.File(path, "w") as f:
            ds = f.create_dataset(
                "fields",
                shape=(T, C, H, W),
                dtype="float32",
                chunks=(1, C, H, W),
                fillvalue=0.0,
            )
            ds.attrs["variables"] = var_names
            ds.attrs["time_step"] = dims["time_step"]
            f.create_dataset("global_means", data=means)
            f.create_dataset("global_stds", data=stds)

        size_kb = os.path.getsize(path) / 1024
        print(f"  {year}.h5  shape=({T},{C},{H},{W})  "
              f"logical={T*C*H*W*4/1024**3:.1f}GB  actual={size_kb:.1f}KB")


if __name__ == "__main__":
    cfg_datapipe = YParams("conf/config.yaml", "datapipe")

    if cfg_datapipe.dataset.data_dir.startswith("/public/") or \
       cfg_datapipe.dataset.data_dir.startswith("/work2/"):
        print("Please check config, ensure data_dir points to local test path "
              "instead of production path.")
        exit()

    years = (
        cfg_datapipe.dataset.train_time +
        cfg_datapipe.dataset.val_time +
        cfg_datapipe.dataset.test_time
    )
    atm_vars = cfg_datapipe.dataset.channels

    generate_fake_h5(cfg_datapipe.dataset.data_dir, atm_vars, years, DATASET_DIMS)

    print("\nFake datasets generated successfully.")
    print(f"  Variables: {len(atm_vars)}")
    print(f"  Years: {years}")
    print(f"  Resolution: {DATASET_DIMS['H']}x{DATASET_DIMS['W']}")
