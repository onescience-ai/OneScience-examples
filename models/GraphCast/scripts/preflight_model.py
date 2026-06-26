#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys

try:
    import h5py
    import numpy as np
    import yaml
except ImportError as exc:
    print(f"缺少依赖: {exc.name}。请先安装 OneScience earth 环境或对应 Python 包。", file=sys.stderr)
    sys.exit(2)


REQUIRED_TOP_LEVEL = ("model", "datapipe")
REQUIRED_MODEL_KEYS = (
    "processor_layers",
    "hidden_dim",
    "mesh_level",
    "num_channels_climate",
    "use_cos_zenith",
    "use_time_of_year_index",
    "num_history",
    "num_channels_static",
    "max_epoch",
    "num_iters_step1",
    "num_iters_step2",
    "num_iters_step3",
    "val_freq",
    "checkpoint_dir",
)
REQUIRED_DATASET_KEYS = (
    "data_dir",
    "dataset_metadata_path",
    "time_diff_std_path",
    "static_dir",
    "train_time",
    "val_time",
    "test_time",
    "channels",
    "img_size",
)
REQUIRED_H5_DATASETS = ("fields", "global_means", "global_stds")
REQUIRED_STATIC_FILES = (
    "land_mask.npy",
    "soil_type.npy",
    "topography.npy",
    "geopotential.nc",
    "land_sea_mask.nc",
)


def fail(message):
    print(f"[FAIL] {message}", file=sys.stderr)
    sys.exit(1)


def ok(message):
    print(f"[OK] {message}")


def load_config(path):
    if not os.path.isfile(path):
        fail(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        fail(f"配置文件不是 YAML mapping: {path}")
    return cfg


def require_keys(mapping, keys, prefix):
    if not isinstance(mapping, dict):
        fail(f"{prefix} 必须是 YAML mapping")
    missing = [key for key in keys if key not in mapping]
    if missing:
        fail(f"{prefix} 缺少字段: {', '.join(missing)}")


def decode_variables(raw_variables):
    return [item.decode() if isinstance(item, bytes) else str(item) for item in raw_variables]


def reject_unsafe_default_path(path, label):
    if os.path.isabs(path) or path.startswith("/public/") or path.startswith("/work"):
        fail(f"{label} 应指向会话工作区内的相对路径或显式挂载目录，当前默认值不适合标准包自动运行")


def validate_years(dataset_cfg):
    all_years = []
    for key in ("train_time", "val_time", "test_time"):
        years = dataset_cfg[key]
        if not isinstance(years, list) or not years:
            fail(f"datapipe.dataset.{key} 必须是非空年份列表")
        for year in years:
            if not isinstance(year, int):
                fail(f"datapipe.dataset.{key} 只能包含整数年份")
        all_years.extend(years)
    return all_years


def inspect_h5_file(path, channels, expected_shape_hw):
    with h5py.File(path, "r") as f:
        for key in REQUIRED_H5_DATASETS:
            if key not in f:
                fail(f"{path} 缺少 HDF5 数据集: {key}")
        fields = f["fields"]
        if len(fields.shape) != 4:
            fail(f"{path}: fields 维度应为 [T,C,H,W]，实际为 {fields.shape}")

        _, channel_count, height, width = fields.shape
        if (height, width) != expected_shape_hw:
            fail(f"{path}: fields 空间尺寸 {(height, width)} 与配置 {expected_shape_hw} 不一致")
        if channel_count < len(channels):
            fail(f"{path}: fields 通道数 {channel_count} 小于配置通道数 {len(channels)}")

        raw_variables = fields.attrs.get("variables")
        if raw_variables is None:
            fail(f"{path}: fields.attrs 缺少 variables")
        variables = decode_variables(raw_variables)
        missing_channels = [channel for channel in channels if channel not in variables]
        if missing_channels:
            fail(f"{path}: HDF5 variables 缺少配置通道: {', '.join(missing_channels)}")
        if "time_step" not in fields.attrs:
            fail(f"{path}: fields.attrs 缺少 time_step")

        for key in ("global_means", "global_stds"):
            arr = f[key]
            if len(arr.shape) != 4:
                fail(f"{path}: {key} 维度应为 [1,C,1,1]，实际为 {arr.shape}")
            if arr.shape[1] < len(channels):
                fail(f"{path}: {key} 通道数 {arr.shape[1]} 小于配置通道数 {len(channels)}")


def inspect_metadata(path, channels):
    if not os.path.isfile(path):
        fail(f"GraphCast 变量元数据文件不存在: {path}。请先运行 python get_data_json.py")
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    channel_mapping = metadata.get("coords", {}).get("channel", {})
    if not isinstance(channel_mapping, dict):
        fail(f"{path}: 缺少 coords.channel mapping")
    values = [str(value) for _, value in sorted(channel_mapping.items(), key=lambda item: int(item[0]))]
    missing = [channel for channel in channels if channel not in values]
    if missing:
        fail(f"{path}: coords.channel 缺少配置通道: {', '.join(missing)}")
    ok(f"GraphCast 变量元数据可读: {path}")


def inspect_time_diff_std(path, channel_count):
    if not os.path.isfile(path):
        fail(f"GraphCast time_diff_std 文件不存在: {path}。请先运行 python compute_time_diff_std.py")
    arr = np.load(path)
    if arr.ndim not in (1, 2):
        fail(f"{path}: time_diff_std 维度应为 [C] 或 [B,C]，实际为 {arr.shape}")
    observed_channels = arr.shape[-1]
    if observed_channels != channel_count:
        fail(f"{path}: time_diff_std 通道数 {observed_channels} 与配置通道数 {channel_count} 不一致")
    if not np.all(np.isfinite(arr)):
        fail(f"{path}: time_diff_std 包含 NaN 或 Inf")
    ok(f"GraphCast time_diff_std 可读: {path}")


def inspect_static_dir(path):
    if not os.path.isdir(path):
        fail(f"GraphCast 静态文件目录不存在: {path}")
    missing = [name for name in REQUIRED_STATIC_FILES if not os.path.isfile(os.path.join(path, name))]
    if missing:
        fail(f"{path} 缺少静态文件: {', '.join(missing)}")
    for name in ("land_mask.npy", "soil_type.npy", "topography.npy"):
        arr = np.load(os.path.join(path, name))
        if arr.shape != (721, 1440):
            fail(f"{os.path.join(path, name)} shape 应为 (721, 1440)，实际为 {arr.shape}")
    ok(f"GraphCast 静态文件可读: {path}")


def main():
    parser = argparse.ArgumentParser(description="GraphCast 模型标准包预检")
    parser.add_argument("--config", default="conf/config.yaml", help="模型配置文件路径")
    parser.add_argument("--check-data", action="store_true", help="检查 data_dir 下已下载 ERA5 HDF5、静态文件和派生统计文件")
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_keys(cfg, REQUIRED_TOP_LEVEL, "配置根节点")
    require_keys(cfg["model"], REQUIRED_MODEL_KEYS, "model")
    require_keys(cfg["datapipe"].get("dataset", {}), REQUIRED_DATASET_KEYS, "datapipe.dataset")

    model_cfg = cfg["model"]
    dataset_cfg = cfg["datapipe"]["dataset"]
    channels = dataset_cfg["channels"]
    if not isinstance(channels, list) or not channels:
        fail("datapipe.dataset.channels 必须是非空列表")
    if int(model_cfg["num_channels_climate"]) != len(channels):
        fail("model.num_channels_climate 必须等于 datapipe.dataset.channels 长度")

    img_size = dataset_cfg["img_size"]
    if not isinstance(img_size, list) or len(img_size) != 2:
        fail("GraphCast 的 datapipe.dataset.img_size 必须是 [H,W]")
    expected_shape_hw = (int(img_size[0]), int(img_size[1]))

    data_dir = dataset_cfg["data_dir"]
    dataset_metadata_path = dataset_cfg["dataset_metadata_path"]
    time_diff_std_path = dataset_cfg["time_diff_std_path"]
    static_dir = dataset_cfg["static_dir"]
    checkpoint_dir = model_cfg["checkpoint_dir"]
    for label, path in (
        ("datapipe.dataset.data_dir", data_dir),
        ("datapipe.dataset.dataset_metadata_path", dataset_metadata_path),
        ("datapipe.dataset.time_diff_std_path", time_diff_std_path),
        ("datapipe.dataset.static_dir", static_dir),
        ("model.checkpoint_dir", checkpoint_dir),
    ):
        reject_unsafe_default_path(path, label)

    all_years = validate_years(dataset_cfg)

    ok(f"配置可解析: {args.config}")
    ok(f"配置通道数: {len(channels)}")
    ok(f"训练/验证/测试年份: {all_years}")
    ok(f"默认数据目录: {data_dir}")
    ok(f"静态文件目录: {static_dir}")
    ok(f"checkpoint 目录: {checkpoint_dir}")
    ok(f"GraphCast 派生文件路径: {dataset_metadata_path}, {time_diff_std_path}")

    if not args.check_data:
        ok("未启用 --check-data，仅完成模型配置级预检")
        return

    h5_paths = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    if not h5_paths:
        fail(f"未找到数据文件: {os.path.join(data_dir, 'data', '*.h5')}")

    years_present = {int(os.path.splitext(os.path.basename(path))[0]) for path in h5_paths}
    missing_years = sorted(set(all_years) - years_present)
    if missing_years:
        fail(f"缺少年份 HDF5 文件: {missing_years}")

    for h5_path in h5_paths:
        inspect_h5_file(h5_path, channels, expected_shape_hw)
    ok(f"HDF5 数据结构可读: {len(h5_paths)} 个文件")

    inspect_static_dir(static_dir)
    inspect_metadata(dataset_metadata_path, channels)
    inspect_time_diff_std(time_diff_std_path, len(channels))

    for name in ("model_bak.pth", "model_finetune_bak.pth"):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            ok(f"发现 checkpoint: {path}")
        else:
            ok(f"未发现 {name}；训练、微调或推理前需要生成/放置该文件")


if __name__ == "__main__":
    main()
