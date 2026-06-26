#!/usr/bin/env python3
"""Validate the AlphaGenome ModelScope package metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import yaml


EXPECTED_REPO_ID = "OneScience/AlphaGenome"
REQUIRED_README_SECTIONS = [
    "## OneScience 官方信息",
    "## 项目说明",
    "## Resource Card",
    "## 文件说明",
    "## Manifest",
    "## 模型 vs 数据集关系",
    "## 文件与下载",
    "## 环境安装",
    "## 运行流程",
    "## 输出说明",
    "## 预检与诊断",
    "## 限制与适用范围",
    "## 引用与许可证",
]


def iter_values(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_values(value)
    else:
        yield obj


def collect_command_names(commands):
    names = set()
    download = commands.get("download", [])
    if isinstance(download, list):
        for item in download:
            if isinstance(item, dict) and item.get("name"):
                names.add(item["name"])
    for key, value in commands.items():
        if key == "download":
            continue
        if isinstance(value, dict) and value.get("name"):
            names.add(value["name"])
    return names


def collect_command_refs(obj):
    refs = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "command_ref":
                refs.append(value)
            else:
                refs.extend(collect_command_refs(value))
    elif isinstance(obj, list):
        for value in obj:
            refs.extend(collect_command_refs(value))
    return refs


def cjk_count(text):
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def validate(root: Path) -> list[str]:
    errors = []
    readme_path = root / "README.md"
    manifest_path = root / "manifest.yaml"

    try:
        readme_bytes = readme_path.read_bytes()
        readme = readme_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"README.md is not valid UTF-8: {exc}"]

    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [f"manifest.yaml is not valid UTF-8: {exc}"]

    manifest = yaml.safe_load(manifest_text)

    for section in REQUIRED_README_SECTIONS:
        if section not in readme:
            errors.append(f"missing README section: {section}")

    for label, text in [("README.md", readme), ("manifest.yaml", manifest_text)]:
        if "????" in text:
            errors.append(f"{label} contains four consecutive question marks")
        if "\ufffd" in text:
            errors.append(f"{label} contains U+FFFD replacement character")
        if cjk_count(text) < (100 if label == "README.md" else 20):
            errors.append(f"{label} CJK count is unexpectedly low: {cjk_count(text)}")

    if manifest.get("resource", {}).get("id") != EXPECTED_REPO_ID:
        errors.append("resource.id does not match OneScience/AlphaGenome")
    primary = manifest.get("platform_resource", {}).get("primary", {})
    if primary.get("repo_id") != EXPECTED_REPO_ID:
        errors.append("platform_resource.primary.repo_id does not match")
    if primary.get("repo_type") != "model":
        errors.append("platform_resource.primary.repo_type must be model")

    if "OneScience/AlphaGenome" not in readme:
        errors.append("README.md does not mention expected repo id")
    if re.search(r"(?<!OneScience/)AlphaGenome(?![\w/-])", readme):
        # The model name is allowed, but download/repo references must be namespaced.
        pass

    repo_like_keys = {"repo_id", "resource_url", "url", "command"}

    def walk_repo_like(obj, parent_key=None):
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield from walk_repo_like(value, key)
        elif isinstance(obj, list):
            for value in obj:
                yield from walk_repo_like(value, parent_key)
        else:
            if parent_key in repo_like_keys:
                yield obj

    for value in walk_repo_like(manifest):
        if isinstance(value, str):
            if "Onescience/" in value or "onescience/" in value:
                errors.append(f"bad namespace casing in manifest value: {value}")
            if value.startswith("https://modelscope.cn/models/") and EXPECTED_REPO_ID not in value:
                errors.append(f"unexpected ModelScope model URL: {value}")
    if "Onescience/" in readme or "onescience/" in readme:
        errors.append("README contains bad namespace casing")

    command_names = collect_command_names(manifest.get("commands", {}))
    refs = collect_command_refs(manifest)
    for ref in refs:
        if ref not in command_names:
            errors.append(f"command_ref has no matching command: {ref}")

    relations = manifest.get("relations", {})
    if not isinstance(relations.get("required_datasets"), list):
        errors.append("relations.required_datasets must be a list")
    if relations.get("required_datasets") != []:
        errors.append("required_datasets should be empty for this package")

    required_paths = [
        "run_inference.py",
        "run_variant_scoring.py",
        "run_track_prediction_eval.py",
        "preflight_alphagenome.py",
        "checkpoints/alphagenome-all-folds/_CHECKPOINT_METADATA",
        "checkpoints/alphagenome-all-folds/manifest.ocdbt",
    ]
    for rel in required_paths:
        if not (root / rel).exists():
            errors.append(f"missing required file: {rel}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    errors = validate(Path(args.root).resolve())
    if errors:
        for error in errors:
            print("ERROR:", error, file=sys.stderr)
        return 1
    print("PACKAGE VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
