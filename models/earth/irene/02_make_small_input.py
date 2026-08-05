from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "examples" / "sample_data.nc"
OUTPUT = ROOT / "examples" / "sample_small_6x64.nc"

TIME_STEPS = 6
CROP_SIZE = 64


def choose_variable(ds: xr.Dataset) -> str:
    if "RR" in ds.data_vars:
        return "RR"
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise KeyError(
        f"NetCDF 中没有 RR，且包含多个变量，无法自动判断: {list(ds.data_vars)}"
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"找不到示例数据: {SOURCE}")

    print(f"读取: {SOURCE}")

    with xr.open_dataset(SOURCE) as ds:
        variable = choose_variable(ds)
        radar = ds[variable]

        if radar.ndim != 3:
            raise ValueError(
                f"输入必须为三维数据，当前 dims={radar.dims}, shape={radar.shape}"
            )

        time_dim = "time" if "time" in radar.dims else radar.dims[0]
        spatial_dims = [dim for dim in radar.dims if dim != time_dim]

        if len(spatial_dims) != 2:
            raise ValueError(f"无法识别两个空间维度，当前维度为: {radar.dims}")

        radar = radar.transpose(time_dim, spatial_dims[0], spatial_dims[1])

        if radar.sizes[time_dim] < TIME_STEPS:
            raise ValueError(
                f"至少需要 {TIME_STEPS} 帧，当前只有 {radar.sizes[time_dim]} 帧。"
            )

        indexers: dict[str, slice] = {time_dim: slice(0, TIME_STEPS)}

        for dim in spatial_dims:
            length = radar.sizes[dim]
            if length < CROP_SIZE:
                raise ValueError(
                    f"空间维度 {dim} 长度为 {length}，小于裁剪尺寸 {CROP_SIZE}。"
                )
            start = (length - CROP_SIZE) // 2
            indexers[dim] = slice(start, start + CROP_SIZE)

        small = radar.isel(indexers).astype(np.float32).load()
        small.name = "RR"
        small.attrs = dict(radar.attrs)
        small.attrs["source_variable"] = variable
        small.attrs["description"] = "IRENE smoke-test input: 6 frames, 64x64 center crop"
        small.encoding = {}

    output_ds = xr.Dataset({"RR": small})
    output_ds.attrs["source_file"] = str(SOURCE)
    output_ds.attrs["purpose"] = "IRENE small inference smoke test"
    output_ds.to_netcdf(OUTPUT)

    print("=" * 72)
    print("小数据制作完成")
    print(f"输出文件: {OUTPUT}")
    print(f"变量: RR")
    print(f"维度: {small.dims}")
    print(f"形状: {small.shape}")
    print(f"最小值: {float(np.nanmin(small.values)):.6f}")
    print(f"最大值: {float(np.nanmax(small.values)):.6f}")
    print(f"NaN 数量: {int(np.isnan(small.values).sum())}")
    print("=" * 72)

    expected = (TIME_STEPS, CROP_SIZE, CROP_SIZE)
    if small.shape != expected:
        raise RuntimeError(f"输出形状错误：实际 {small.shape}，预期 {expected}")


if __name__ == "__main__":
    main()
