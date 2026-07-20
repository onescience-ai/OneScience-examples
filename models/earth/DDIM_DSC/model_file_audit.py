#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读扫描模型目录，列出不适合提交到 Git 的文件候选。

脚本不会上传、下载、备份、重命名、删除或修改任何文件。

默认显示四类内容：
1. 大于等于指定阈值的普通文件（默认 1 MiB）；
2. 常见模型、数据和压缩文件，例如 .npy、.npz、.h5、.pt；
3. 常见无用文件和目录，例如 .ipynb、__pycache__；
4. 其他隐藏文件和隐藏目录，供人工判断。

示例：
    python model_file_audit.py --model-dir /data/upmodels/my_model
    python model_file_audit.py --model-dir /data/upmodels/my_model --threshold-mb 100
    python model_file_audit.py /data/upmodels/my_model --absolute-paths
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


MIB = 1024 * 1024

# 命中这些目录后只显示目录本身，不再展开其中的文件，避免刷屏。
JUNK_DIR_NAMES = {
    "__pycache__",
    ".cache",
    ".hypothesis",
    ".ipynb_checkpoints",
    ".ms_upload_cache",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
}

JUNK_FILE_NAMES = {
    ".coverage",
    ".DS_Store",
    "desktop.ini",
    "Thumbs.db",
}

JUNK_FILE_SUFFIXES = {
    ".ipynb",
    ".pyc",
    ".pyo",
    ".swp",
    ".swo",
}

# 这些文件无论大小都会显示，便于人工判断是否应保留在 Git 仓库中。
MODEL_DATA_FILE_SUFFIXES = {
    # NumPy、科学计算和通用序列化
    ".h5",
    ".hdf5",
    ".joblib",
    ".mat",
    ".msgpack",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    # 常见模型、权重和推理格式
    ".bin",
    ".ckpt",
    ".model",
    ".onnx",
    ".ot",
    ".pb",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
    # 表格、数据集和数据库
    ".arrow",
    ".csv",
    ".db",
    ".feather",
    ".json",
    ".jsonl",
    ".mdb",
    ".ndjson",
    ".parquet",
    ".sqlite",
    ".sqlite3",
    ".tsv",
    # 压缩包和归档文件
    ".7z",
    ".bz2",
    ".gz",
    ".rar",
    ".tar",
    ".xz",
    ".zip",
}

# .git 是仓库元数据，不属于待提交内容；扫描它既无意义又会产生大量噪声。
ALWAYS_SKIP_DIR_NAMES = {".git"}


@dataclass(frozen=True)
class LargeFile:
    path: Path
    size: int


@dataclass(frozen=True)
class ScanResult:
    large_files: list[LargeFile]
    model_data_files: list[LargeFile]
    junk_paths: list[Path]
    hidden_paths: list[Path]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读列出模型目录中的大文件、模型/数据文件、常见无用项和其他隐藏项。"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        help="要扫描的模型目录；也可以使用 --model-dir",
    )
    parser.add_argument(
        "--model-dir",
        help="要扫描的模型目录（兼容原脚本的参数形式）",
    )
    parser.add_argument(
        "--threshold-mb",
        type=float,
        default=0.5,
        metavar="MB",
        help="大文件阈值，单位 MiB，默认 0.5；例如 Git 常用检查可设为 100",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="显示绝对路径；默认显示相对于模型目录的路径",
    )
    return parser.parse_args()


def resolve_model_dir(args: argparse.Namespace) -> Path:
    if args.directory and args.model_dir:
        raise ValueError("位置参数和 --model-dir 只能使用一个。")

    raw_path = args.model_dir or args.directory
    if not raw_path:
        raise ValueError("请指定模型目录，例如：--model-dir /data/upmodels/my_model")
    if args.threshold_mb <= 0:
        raise ValueError("--threshold-mb 必须大于 0。")

    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"目录不存在: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"目标不是目录: {path}")
    return path


def is_hidden_name(name: str) -> bool:
    return name.startswith(".") and name not in {".", ".."}


def is_junk_file(path: Path) -> bool:
    return path.name in JUNK_FILE_NAMES or path.suffix.lower() in JUNK_FILE_SUFFIXES


def scan_model_dir(root: Path, threshold_bytes: int) -> ScanResult:
    large_files: list[LargeFile] = []
    model_data_files: list[LargeFile] = []
    junk_paths: list[Path] = []
    hidden_paths: list[Path] = []
    warnings: list[str] = []

    def on_walk_error(exc: OSError) -> None:
        location = exc.filename or "未知路径"
        warnings.append(f"无法读取 {location}: {exc.strerror or exc}")

    for current_raw, dir_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=on_walk_error,
    ):
        current = Path(current_raw)
        dirs_to_visit: list[str] = []

        for name in dir_names:
            path = current / name
            if name in ALWAYS_SKIP_DIR_NAMES:
                continue
            if name in JUNK_DIR_NAMES:
                junk_paths.append(path)
                continue
            if is_hidden_name(name):
                hidden_paths.append(path)
                continue
            dirs_to_visit.append(name)

        # 原地修改才会让 os.walk 跳过已归类的目录。
        dir_names[:] = dirs_to_visit

        for name in file_names:
            path = current / name

            if is_junk_file(path):
                junk_paths.append(path)
                continue
            if is_hidden_name(name):
                hidden_paths.append(path)
                continue
            if path.is_symlink():
                # Git 记录的是符号链接本身，不会提交链接目标的大文件内容。
                continue

            try:
                size = path.stat().st_size
            except OSError as exc:
                warnings.append(f"无法读取文件大小 {path}: {exc}")
                continue

            if path.suffix.lower() in MODEL_DATA_FILE_SUFFIXES:
                model_data_files.append(LargeFile(path=path, size=size))

            if size >= threshold_bytes:
                large_files.append(LargeFile(path=path, size=size))

    large_files.sort(key=lambda item: (-item.size, item.path.as_posix().casefold()))
    model_data_files.sort(key=lambda item: (-item.size, item.path.as_posix().casefold()))
    junk_paths.sort(key=lambda path: path.as_posix().casefold())
    hidden_paths.sort(key=lambda path: path.as_posix().casefold())
    return ScanResult(
        large_files=large_files,
        model_data_files=model_data_files,
        junk_paths=junk_paths,
        hidden_paths=hidden_paths,
        warnings=warnings,
    )


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def display_path(path: Path, root: Path, absolute_paths: bool) -> str:
    return str(path) if absolute_paths else path.relative_to(root).as_posix()


def print_report(
    root: Path,
    threshold_bytes: int,
    result: ScanResult,
    absolute_paths: bool,
) -> None:
    print("模型目录只读检查")
    print(f"扫描目录: {root}")
    print(f"大文件阈值: {human_size(threshold_bytes)}")
    print("说明: 本脚本只显示候选项，不会上传、删除或修改任何文件。")

    total_large_size = sum(item.size for item in result.large_files)
    print(
        f"\n[1/4] 大文件: {len(result.large_files)} 个，"
        f"合计 {human_size(total_large_size)}"
    )
    if result.large_files:
        for item in result.large_files:
            path_text = display_path(item.path, root, absolute_paths)
            print(f"  {human_size(item.size):>11}  {path_text}")
    else:
        print("  未发现。")

    total_model_data_size = sum(item.size for item in result.model_data_files)
    print(
        f"\n[2/4] 常见模型、数据或压缩文件: {len(result.model_data_files)} 个，"
        f"合计 {human_size(total_model_data_size)}"
    )
    print("  提示: 此分组与大文件分组可能重叠。")
    if result.model_data_files:
        for item in result.model_data_files:
            path_text = display_path(item.path, root, absolute_paths)
            print(f"  {human_size(item.size):>11}  {path_text}")
    else:
        print("  未发现。")

    print(f"\n[3/4] 常见无用文件或目录: {len(result.junk_paths)} 项")
    if result.junk_paths:
        for path in result.junk_paths:
            kind = "目录" if path.is_dir() else "文件"
            print(f"  [{kind}] {display_path(path, root, absolute_paths)}")
    else:
        print("  未发现。")

    print(f"\n[4/4] 其他隐藏文件或目录（请人工判断）: {len(result.hidden_paths)} 项")
    if result.hidden_paths:
        for path in result.hidden_paths:
            kind = "目录" if path.is_dir() else "文件"
            print(f"  [{kind}] {display_path(path, root, absolute_paths)}")
    else:
        print("  未发现。")

    candidate_paths = {
        *(item.path for item in result.large_files),
        *(item.path for item in result.model_data_files),
        *result.junk_paths,
        *result.hidden_paths,
    }
    candidate_count = len(candidate_paths)
    print(f"\n检查完成，共显示 {candidate_count} 项候选内容。请人工确认后再处理。")

    if result.warnings:
        print(f"\n警告: 有 {len(result.warnings)} 个路径未能完整检查:", file=sys.stderr)
        for warning in result.warnings:
            print(f"  {warning}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    try:
        model_dir = resolve_model_dir(args)
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    threshold_bytes = max(1, int(args.threshold_mb * MIB))
    result = scan_model_dir(model_dir, threshold_bytes)
    print_report(
        root=model_dir,
        threshold_bytes=threshold_bytes,
        result=result,
        absolute_paths=args.absolute_paths,
    )
    return 0 if not result.warnings else 1


if __name__ == "__main__":
    raise SystemExit(main())
