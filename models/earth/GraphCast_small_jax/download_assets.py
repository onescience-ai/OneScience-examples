#!/usr/bin/env python3
"""List and download the minimal real GraphCast_small assets anonymously."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

BUCKET_NAME = "dm_graphcast"
STAT_OBJECTS = (
    "stats/diffs_stddev_by_level.nc",
    "stats/mean_by_level.nc",
    "stats/stddev_by_level.nc",
)


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def anonymous_bucket():
    from google.cloud import storage

    client = storage.Client.create_anonymous_client()
    return client.bucket(BUCKET_NAME)


def list_objects(bucket, prefixes: Iterable[str]) -> list[dict[str, Any]]:
    result = []
    for prefix in prefixes:
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name == prefix:
                continue
            result.append({
                "name": blob.name,
                "size": int(blob.size or 0),
                "md5_hash": blob.md5_hash,
                "updated": blob.updated.isoformat() if blob.updated else None,
            })
    return result


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def md5_base64_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"bucket": BUCKET_NAME, "files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data.get("files"), dict):
            raise ValueError("manifest files entry is not an object")
        return data
    except Exception as exc:
        raise RuntimeError(f"Cannot read manifest {path}: {exc}") from exc


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def select_defaults(objects: list[dict[str, Any]], dataset_override: str | None) -> list[dict[str, Any]]:
    params = [item for item in objects if item["name"].startswith("params/GraphCast_small")]
    if len(params) != 1:
        raise RuntimeError(f"Expected one GraphCast_small checkpoint, found {len(params)}")

    if dataset_override:
        wanted = dataset_override
        if not wanted.startswith("dataset/"):
            wanted = "dataset/" + wanted
        datasets = [item for item in objects if item["name"] == wanted]
    else:
        datasets = [
            item for item in objects
            if item["name"].startswith("dataset/source-era5_")
            and "_res-1.0_" in item["name"]
            and "_levels-13_" in item["name"]
            and "_steps-01.nc" in item["name"]
        ]
    if len(datasets) != 1:
        raise RuntimeError(
            "Expected one matching ERA5 1-degree/13-level/1-step dataset, "
            f"found {len(datasets)}"
        )

    stats = [item for item in objects if item["name"] in STAT_OBJECTS]
    if len(stats) != len(STAT_OBJECTS):
        raise RuntimeError(f"Expected {len(STAT_OBJECTS)} stats files, found {len(stats)}")
    return params + datasets + sorted(stats, key=lambda item: item["name"])


def manual_url(object_name: str) -> str:
    return f"https://storage.googleapis.com/{BUCKET_NAME}/{quote(object_name)}"


def download_one(bucket, item: dict[str, Any], assets_dir: Path, manifest: dict[str, Any]) -> None:
    object_name = item["name"]
    expected_size = int(item["size"])
    expected_md5 = item.get("md5_hash")
    destination = assets_dir / Path(object_name)
    part = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    previous = manifest["files"].get(object_name, {})
    if destination.exists() and destination.stat().st_size == expected_size:
        digest = sha256_file(destination)
        md5_ok = not expected_md5 or md5_base64_file(destination) == expected_md5
        if previous.get("sha256") in (None, digest) and md5_ok:
            print(
                f"SKIP verified: {object_name} ({human_size(expected_size)}), "
                f"SHA256={digest}"
            )
            manifest["files"][object_name] = {
                "size": expected_size,
                "remote_md5_base64": expected_md5,
                "sha256": digest,
                "path": str(destination),
            }
            return
        print(f"Existing file failed checksum; downloading again: {destination}")

    offset = part.stat().st_size if part.exists() else 0
    if offset > expected_size:
        print(f"Discarding oversized partial file: {part} ({offset} bytes)")
        part.unlink()
        offset = 0

    print(
        f"DOWNLOAD: {object_name}\n"
        f"  remote size: {expected_size} bytes ({human_size(expected_size)})\n"
        f"  local partial size before: {offset} bytes ({human_size(offset)})"
    )
    if offset < expected_size:
        blob = bucket.blob(object_name)
        mode = "ab" if offset else "wb"
        with part.open(mode) as stream:
            blob.download_to_file(
                stream, start=offset if offset else None, raw_download=True
            )

    actual_part_size = part.stat().st_size
    print(f"  local size after transfer: {actual_part_size} bytes ({human_size(actual_part_size)})")
    if actual_part_size != expected_size:
        raise RuntimeError(
            f"Incomplete download for {object_name}: expected {expected_size}, got {actual_part_size}. "
            "The .part file was retained for resume."
        )
    if expected_md5:
        actual_md5 = md5_base64_file(part)
        if actual_md5 != expected_md5:
            part.unlink()
            raise RuntimeError(
                f"MD5 mismatch for {object_name}: expected {expected_md5}, got {actual_md5}"
            )
    digest = sha256_file(part)
    part.replace(destination)
    manifest["files"][object_name] = {
        "size": expected_size,
        "remote_md5_base64": expected_md5,
        "sha256": digest,
        "path": str(destination),
    }
    print(f"  verified SHA256: {digest}")


def main() -> int:
    default_root = Path(os.environ.get(
        "WEATHER_DATA_ROOT", "/root/group_data/SDU-Test/weather"
    ))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=default_root / "assets")
    parser.add_argument("--list", action="store_true", help="List params, dataset and stats objects")
    parser.add_argument("--dataset", help="Override the default compatible dataset object/name")
    args = parser.parse_args()

    try:
        bucket = anonymous_bucket()
        objects = list_objects(bucket, ("params/", "dataset/", "stats/"))
    except Exception as exc:
        print(f"ERROR: anonymous access to gs://{BUCKET_NAME} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Manual minimum download list:", file=sys.stderr)
        for name in (
            "params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
            "dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc",
            *STAT_OBJECTS,
        ):
            print(f"  {manual_url(name)}", file=sys.stderr)
        return 2

    if args.list:
        for item in objects:
            print(f"{item['name']}\t{item['size']} bytes\t{human_size(item['size'])}")
        return 0

    selected = select_defaults(objects, args.dataset)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.assets_dir / "assets_manifest.json"
    manifest = load_manifest(manifest_path)
    manifest["selection"] = [item["name"] for item in selected]
    print("Selected files:")
    for item in selected:
        print(f"  {item['name']} ({human_size(item['size'])})")
    try:
        for item in selected:
            download_one(bucket, item, args.assets_dir, manifest)
            save_manifest(manifest_path, manifest)
    except Exception as exc:
        save_manifest(manifest_path, manifest)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Manual download URLs for selected files:", file=sys.stderr)
        for item in selected:
            print(f"  {manual_url(item['name'])}", file=sys.stderr)
        return 2
    print(f"Asset manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
