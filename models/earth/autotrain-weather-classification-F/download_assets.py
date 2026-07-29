#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise SystemExit(
        "缺少 huggingface_hub，请先执行：\n"
        f"{sys.executable} -m pip install huggingface_hub==0.36.0"
    ) from exc

MODEL_REPO = "dazzle-nu/autotrain-weather-classification-3739699407"
DATASET_REPO = "dazzle-nu/autotrain-data-weather-classification"

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
DATASET_DIR = PROJECT_DIR / "weather-classification-data"
CACHE_DIR = PROJECT_DIR / ".hf_cache"

FILES = [
    {
        "name": "模型权重",
        "repo_id": MODEL_REPO,
        "repo_type": "model",
        "filename": "pytorch_model.bin",
        "local_dir": MODEL_DIR,
        "expected_min_bytes": 300 * 1024 * 1024,
    },
    {
        "name": "验证集数据",
        "repo_id": DATASET_REPO,
        "repo_type": "dataset",
        "filename": "processed/valid/dataset.arrow",
        "local_dir": DATASET_DIR,
        "expected_min_bytes": 120 * 1024 * 1024,
    },
]

def format_size(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 2):.2f} MiB"

def download_file(item: dict) -> Path:
    item["local_dir"].mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    target_path = item["local_dir"] / item["filename"]

    if target_path.exists() and target_path.stat().st_size >= item["expected_min_bytes"]:
        print(
            f"[跳过] {item['name']}已存在："
            f"{target_path} ({format_size(target_path.stat().st_size)})"
        )
        return target_path

    print(f"\n[下载] {item['name']}")
    print(f"仓库：{item['repo_id']}")
    print(f"文件：{item['filename']}")
    print(f"保存目录：{item['local_dir']}")

    downloaded_path = hf_hub_download(
        repo_id=item["repo_id"],
        repo_type=item["repo_type"],
        filename=item["filename"],
        revision="main",
        local_dir=str(item["local_dir"]),
        cache_dir=str(CACHE_DIR),
        force_download=False,
    )

    target_path = Path(downloaded_path)

    if not target_path.exists():
        raise FileNotFoundError(f"下载完成后未找到文件：{target_path}")

    actual_size = target_path.stat().st_size
    if actual_size < item["expected_min_bytes"]:
        raise RuntimeError(
            f"文件大小异常：{target_path}，实际 {format_size(actual_size)}"
        )

    print(f"[完成] {target_path} ({format_size(actual_size)})")
    return target_path

def main() -> None:
    print("=" * 72)
    print("天气分类模型大文件下载")
    print("=" * 72)
    print(f"项目目录：{PROJECT_DIR}")

    downloaded = [download_file(item) for item in FILES]

    print("\n" + "=" * 72)
    print("全部大文件检查完成")
    print("=" * 72)

    for path in downloaded:
        print(f"{format_size(path.stat().st_size):>12}  {path}")

if __name__ == "__main__":
    main()
