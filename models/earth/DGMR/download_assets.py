#!/usr/bin/env python3
"""Download the source code and pretrained checkpoint required by test.py."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SOURCE_REPOSITORY = "https://github.com/openclimatefix/skillful_nowcasting.git"
MODEL_REPOSITORY = "openclimatefix/dgmr"
MODEL_REVISION = "af3ebd96bfe4d3f5e427feff7aac71748044e086"
MODEL_FILES = ("config.json", "pytorch_model.bin")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_source_commit() -> str:
    path = ROOT / "source_commit.txt"
    commit = path.read_text(encoding="utf-8").strip()

    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"Invalid source commit in {path}: {commit!r}")

    return commit


def download_source(source_dir: Path, commit: str) -> None:
    git_dir = source_dir / ".git"

    if source_dir.exists() and not git_dir.exists():
        if any(source_dir.iterdir()):
            raise RuntimeError(
                f"{source_dir} exists, is not empty, and is not a Git repository."
            )
        source_dir.rmdir()

    if not git_dir.exists():
        source_dir.mkdir(parents=True, exist_ok=False)
        run(["git", "-C", str(source_dir), "init", "-q"])
        run(
            [
                "git",
                "-C",
                str(source_dir),
                "remote",
                "add",
                "origin",
                SOURCE_REPOSITORY,
            ]
        )
    else:
        status = subprocess.run(
            ["git", "-C", str(source_dir), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

        if status.strip():
            raise RuntimeError(f"Source repository has local changes: {source_dir}")

    current_commit = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if current_commit != commit:
        run(
            [
                "git",
                "-C",
                str(source_dir),
                "fetch",
                "--depth",
                "1",
                "origin",
                commit,
            ]
        )
        run(["git", "-C", str(source_dir), "checkout", "--detach", "FETCH_HEAD"])

    actual_commit = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if actual_commit != commit:
        raise RuntimeError(
            f"Source commit mismatch: expected {commit}, got {actual_commit}"
        )

    required_source_files = (
        source_dir / "dgmr" / "__init__.py",
        source_dir / "dgmr" / "dgmr.py",
    )

    for path in required_source_files:
        if not path.is_file():
            raise FileNotFoundError(path)


def download_model(model_dir: Path, force: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing huggingface_hub. Install it with: "
            "pip install huggingface-hub"
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        print(f"Downloading {MODEL_REPOSITORY}/{filename}", flush=True)
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=filename,
            revision=MODEL_REVISION,
            local_dir=str(model_dir),
            force_download=force,
        )

    config = model_dir / "config.json"
    checkpoint = model_dir / "pytorch_model.bin"

    if not config.is_file() or config.stat().st_size == 0:
        raise RuntimeError(f"Invalid config file: {config}")

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if checkpoint.stat().st_size < 300_000_000:
        raise RuntimeError(
            f"Checkpoint is unexpectedly small: "
            f"{checkpoint.stat().st_size} bytes"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DGMR source code and pretrained model files."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "skillful_nowcasting",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "pretrained" / "dgmr",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force Hugging Face to download the model files again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    source_commit = read_source_commit()

    print(f"Source repository: {SOURCE_REPOSITORY}")
    print(f"Source commit:     {source_commit}")
    print(f"Source directory:  {source_dir}")
    print(f"Model repository:  {MODEL_REPOSITORY}")
    print(f"Model revision:    {MODEL_REVISION}")
    print(f"Model directory:   {model_dir}")

    download_source(source_dir, source_commit)
    download_model(model_dir, args.force)

    print()
    print("PASS: DGMR assets downloaded and verified.")
    print(f"Source:     {source_dir}")
    print(f"Config:     {model_dir / 'config.json'}")
    print(f"Checkpoint: {model_dir / 'pytorch_model.bin'}")
    print(f"Bytes:      {(model_dir / 'pytorch_model.bin').stat().st_size}")


if __name__ == "__main__":
    main()
