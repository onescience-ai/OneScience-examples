#!/usr/bin/env python3
"""Download TorNet model weights, catalog, and optional 2013 data."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


MODEL_URL = (
    "https://huggingface.co/"
    "tornet-ml/tornado_detector_baseline_v1/"
    "resolve/main/tornado_detector_baseline.keras?download=true"
)

CATALOG_URL = (
    "https://zenodo.org/records/12636522/"
    "files/catalog.csv?download=1"
)

DATA_2013_URL = (
    "https://zenodo.org/records/12636522/"
    "files/tornet_2013.tar.gz?download=1"
)

CATALOG_MD5 = "de0beb814845ba897bbb4e11b0bf563e"
DATA_2013_MD5 = "d924097d0a00744497992aceea79fa1a"


def calculate_md5(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_md5(path: Path, expected: str | None) -> None:
    if expected is None:
        return

    actual = calculate_md5(path)

    if actual != expected:
        raise RuntimeError(
            f"MD5校验失败：{path}\n"
            f"期望值：{expected}\n"
            f"实际值：{actual}"
        )

    print(f"MD5校验通过：{path.name}")


def download_file(
    url: str,
    destination: Path,
    expected_md5: str | None = None,
    force: bool = False,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        destination.name + ".part"
    )

    if force:
        destination.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)

    if destination.exists():
        verify_md5(destination, expected_md5)
        print(f"文件已经存在，跳过：{destination}")
        return

    curl = shutil.which("curl")

    if curl is None:
        raise RuntimeError(
            "当前环境未找到curl命令。"
        )

    print(f"正在下载：{destination.name}")
    print(f"保存到：{destination}")

    command = [
        curl,
        "-fL",
        "-C",
        "-",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "-o",
        str(temporary),
        url,
    ]

    subprocess.run(command, check=True)

    verify_md5(temporary, expected_md5)
    temporary.replace(destination)

    print(f"下载完成：{destination}")


def extract_2013(
    archive: Path,
    output_dir: Path,
) -> None:
    tar = shutil.which("tar")

    if tar is None:
        raise RuntimeError(
            "当前环境未找到tar命令。"
        )

    if not archive.is_file():
        raise FileNotFoundError(
            f"找不到压缩包：{archive}"
        )

    print(f"正在解压：{archive}")

    subprocess.run(
        [
            tar,
            "-xzf",
            str(archive),
            "-C",
            str(output_dir),
        ],
        check=True,
    )

    print(f"解压完成：{output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "下载TorNet预训练模型、catalog.csv，"
            "以及可选的2013年数据。"
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="模型和数据保存目录。",
    )

    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="不下载预训练模型。",
    )

    parser.add_argument(
        "--skip-catalog",
        action="store_true",
        help="不下载catalog.csv。",
    )

    parser.add_argument(
        "--with-2013-data",
        action="store_true",
        help="下载约3.2GB的2013年数据。",
    )

    parser.add_argument(
        "--extract",
        action="store_true",
        help="下载后解压2013年数据。",
    )

    parser.add_argument(
        "--remove-archive",
        action="store_true",
        help="成功解压后删除压缩包。",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载已有文件。",
    )

    args = parser.parse_args()

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.skip_model:
        download_file(
            MODEL_URL,
            output_dir
            / "tornado_detector_baseline.keras",
            force=args.force,
        )

    if not args.skip_catalog:
        download_file(
            CATALOG_URL,
            output_dir / "catalog.csv",
            expected_md5=CATALOG_MD5,
            force=args.force,
        )

    archive = (
        output_dir
        / "tornet_2013.tar.gz"
    )

    if args.with_2013_data:
        download_file(
            DATA_2013_URL,
            archive,
            expected_md5=DATA_2013_MD5,
            force=args.force,
        )

    if args.extract:
        if not archive.exists():
            parser.error(
                "--extract需要本地已有"
                "tornet_2013.tar.gz；"
                "请同时使用--with-2013-data。"
            )

        extract_2013(
            archive,
            output_dir,
        )

        if args.remove_archive:
            archive.unlink()
            print(f"已删除压缩包：{archive}")

    print()
    print("资源准备完成。")
    print(f'export TORNET_ROOT="{output_dir}"')

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\n下载已被用户中断。",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\n错误：{exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
