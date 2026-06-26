#!/usr/bin/env python3
import argparse
import glob
import os
import sys

try:
    import h5py
    import yaml
except ImportError as exc:
    print(f"缺少依赖: {exc.name}。请先安装 OneScience earth 环境或对应 Python 包。", file=sys.stderr)
    sys.exit(2)


REQUIRED_TOP_LEVEL = ("model", "datapipe")
REQUIRED_MODEL_KEYS = ("max_epoch", "checkpoint_dir")
REQUIRED_DATASET_KEYS = ("data_dir", "train_time", "val_time", "test_time", "channels", "img_size")


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
    missing = [key for key in keys if key not in mapping]
    if missing:
        fail(f"{prefix} 缺少字段: {', '.join(missing)}")


def inspect_h5_file(path, channels, expected_height, expected_width):
    with h5py.File(path, "r") as f:
        for key in ("fields", "global_means", "global_stds"):
            if key not in f:
                fail(f"{path} 缺少 HDF5 数据集: {key}")
        fields = f["fields"]
        if len(fields.shape) != 4:
            fail(f"{path}: fields 维度应为 [T,C,H,W]，实际为 {fields.shape}")
        _, channel_count, height, width = fields.shape
        if channel_count < len(channels):
            fail(f"{path}: fields 通道数 {channel_count} 小于配置通道数 {len(channels)}")
        if (height, width) != (expected_height, expected_width):
            fail(f"{path}: fields 空间尺寸 {(height, width)} 与配置 {(expected_height, expected_width)} 不一致")
        raw_variables = fields.attrs.get("variables")
        if raw_variables is None:
            fail(f"{path}: fields.attrs 缺少 variables")
        variables = [v.decode() if isinstance(v, bytes) else str(v) for v in raw_variables]
        missing_channels = [ch for ch in channels if ch not in variables]
        if missing_channels:
            fail(f"{path}: HDF5 variables 缺少配置通道: {', '.join(missing_channels)}")
        if "time_step" not in fields.attrs:
            fail(f"{path}: fields.attrs 缺少 time_step")
        for key in ("global_means", "global_stds"):
            arr = f[key]
            if arr.shape[1] < len(channels):
                fail(f"{path}: {key} 通道数 {arr.shape[1]} 小于配置通道数 {len(channels)}")


def main():
    parser = argparse.ArgumentParser(description="FourCastNet 模型标准包预检")
    parser.add_argument("--config", default="conf/config.yaml", help="模型配置文件路径")
    parser.add_argument("--check-data", action="store_true", help="检查 data_dir 下的 HDF5 数据结构")
    args = parser.parse_args()

    cfg = load_config(args.config)
    require_keys(cfg, REQUIRED_TOP_LEVEL, "配置根节点")

    model_cfg = cfg["model"]
    datapipe_cfg = cfg["datapipe"]
    dataset_cfg = datapipe_cfg.get("dataset", {})
    require_keys(model_cfg, REQUIRED_MODEL_KEYS, "model")
    require_keys(dataset_cfg, REQUIRED_DATASET_KEYS, "datapipe.dataset")

    channels = dataset_cfg["channels"]
    if not isinstance(channels, list) or not channels:
        fail("datapipe.dataset.channels 必须是非空列表")
    img_size = dataset_cfg["img_size"]
    if not isinstance(img_size, list) or len(img_size) != 2:
        fail("datapipe.dataset.img_size 必须是 [H, W]")
    data_dir = dataset_cfg["data_dir"]
    if os.path.isabs(data_dir) or data_dir.startswith("/public/") or data_dir.startswith("/work"):
        fail("datapipe.dataset.data_dir 应指向会话工作区内的相对路径或已挂载数据目录，当前值不适合标准包默认运行")

    all_years = []
    for key in ("train_time", "val_time", "test_time"):
        years = dataset_cfg[key]
        if not isinstance(years, list) or not years:
            fail(f"datapipe.dataset.{key} 必须是非空年份列表")
        all_years.extend(years)

    checkpoint_dir = model_cfg["checkpoint_dir"]
    if os.path.isabs(checkpoint_dir):
        fail("model.checkpoint_dir 应为会话工作区内的相对路径")

    ok(f"配置可解析: {args.config}")
    ok(f"配置通道数: {len(channels)}")
    ok(f"训练/验证/测试年份: {all_years}")
    ok(f"默认数据目录: {data_dir}")
    ok(f"checkpoint 目录: {checkpoint_dir}")

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
        inspect_h5_file(h5_path, channels, int(img_size[0]), int(img_size[1]))
    ok(f"HDF5 数据结构可读: {len(h5_paths)} 个文件")


if __name__ == "__main__":
    main()
