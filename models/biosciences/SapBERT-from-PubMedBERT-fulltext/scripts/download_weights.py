#!/usr/bin/env python3
"""
下载 SapBERT 测试所需的大文件。

下载内容：
1. config/vocab.txt
2. weight/model.safetensors

使用方法：
    python scripts/download_weights.py
"""

import shutil
import sys
from pathlib import Path


REPO_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def download_file(
    hf_hub_download,
    filename: str,
    save_dir: Path,
) -> Path:
    """从 Hugging Face 下载指定文件。"""

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_file = save_dir / filename

    print(f"正在下载：{filename}")
    print(f"保存位置：{target_file}")

    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision="main",
            local_dir=str(save_dir),
        )
    except Exception as exc:
        print(
            f"\n下载失败：{filename}",
            file=sys.stderr,
        )
        print(
            f"错误信息：{exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # 删除 Hugging Face 在目标目录内生成的下载元数据
    local_cache_dir = save_dir / ".cache"

    if local_cache_dir.exists():
        shutil.rmtree(
            local_cache_dir,
            ignore_errors=True,
        )

    if not target_file.is_file():
        print(
            f"\n下载结束后未找到文件：{target_file}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    file_size_mb = (
        target_file.stat().st_size
        / 1024**2
    )

    print(
        f"下载完成：{target_file}"
    )
    print(
        f"文件大小：{file_size_mb:.2f} MB\n"
    )

    return target_file


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "缺少 huggingface_hub。\n"
            "请先执行：\n"
            "python -m pip install -U huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1)

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    config_dir = root_dir / "config"
    weight_dir = root_dir / "weight"

    print("=" * 72)
    print("SapBERT 模型文件下载")
    print("=" * 72)
    print(f"Hugging Face仓库：{REPO_ID}")
    print(f"项目根目录：{root_dir}")
    print()

    download_file(
        hf_hub_download=hf_hub_download,
        filename="vocab.txt",
        save_dir=config_dir,
    )

    download_file(
        hf_hub_download=hf_hub_download,
        filename="model.safetensors",
        save_dir=weight_dir,
    )

    print("=" * 72)
    print("词表和模型权重全部下载完成")
    print("=" * 72)
    print("接下来运行：")
    print("python scripts/test.py")


if __name__ == "__main__":
    main()
