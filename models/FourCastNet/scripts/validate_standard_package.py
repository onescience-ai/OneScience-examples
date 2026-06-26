#!/usr/bin/env python3
import argparse
import os
import re
import sys
from collections import Counter

import yaml


MODEL_ID = "OneScience/FourCastNet/"
DATASET_ID = "OneScience/ERA5"

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
    "## 预检与诊断",
    "## 输出说明",
    "## 限制与适用范围",
    "## 引用与许可证",
]

REQUIRED_MANIFEST_KEYS = [
    "resource",
    "platform_resource",
    "website_integration",
    "runtime",
    "onescience",
    "runtime_package",
    "files",
    "relations",
    "run_matrix",
    "capabilities",
    "commands",
    "expected_outputs",
    "diagnostics",
    "domain_extension",
]


def fail(message):
    print(f"[FAIL] {message}", file=sys.stderr)
    sys.exit(1)


def ok(message):
    print(f"[OK] {message}")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_command_refs(obj):
    refs = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"command_ref", "command_refs"}:
                if isinstance(value, str):
                    refs.append(value)
                elif isinstance(value, list):
                    refs.extend(item for item in value if isinstance(item, str))
            refs.extend(collect_command_refs(value))
    elif isinstance(obj, list):
        for item in obj:
            refs.extend(collect_command_refs(item))
    return refs


def resolve_command_ref(manifest, ref):
    parts = ref.split(".")
    if len(parts) != 3 or parts[0] != "commands":
        return False
    stage, name = parts[1], parts[2]
    commands = manifest.get("commands", {}).get(stage)
    if not isinstance(commands, list):
        return False
    return sum(1 for item in commands if isinstance(item, dict) and item.get("name") == name) == 1


def validate_readme(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    missing = [section for section in REQUIRED_README_SECTIONS if section not in text]
    if missing:
        fail(f"README 缺少章节: {', '.join(missing)}")
    positions = [text.index(section) for section in REQUIRED_README_SECTIONS]
    if positions != sorted(positions):
        fail("README 必需章节顺序不符合标准")
    if MODEL_ID not in text:
        fail(f"README 未包含模型 ID: {MODEL_ID}")
    if DATASET_ID not in text:
        fail(f"README 未包含数据集 ID: {DATASET_ID}")
    ok("README 必需章节齐全且顺序正确")


def validate_manifest(path):
    manifest = load_yaml(path)
    if not isinstance(manifest, dict):
        fail(f"Manifest 不是 YAML mapping: {path}")
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        fail(f"Manifest 缺少关键字段: {', '.join(missing)}")

    if manifest.get("resource_type") != "model":
        fail("resource_type 必须为 model")
    if manifest["resource"].get("id") != MODEL_ID:
        fail(f"resource.id 不等于 {MODEL_ID}")
    primary = manifest["platform_resource"].get("primary", {})
    if primary.get("repo_id") != MODEL_ID:
        fail(f"platform_resource.primary.repo_id 不等于 {MODEL_ID}")
    if primary.get("repo_type") != "model":
        fail("platform_resource.primary.repo_type 必须为 model")

    relations = manifest["relations"]
    required_datasets = relations.get("required_datasets", [])
    matches = [item for item in required_datasets if item.get("id") == DATASET_ID]
    if len(matches) != 1:
        fail(f"relations.required_datasets 必须唯一声明 {DATASET_ID}")
    resource_ref = matches[0].get("resource_ref", {})
    if resource_ref.get("repo_id") != DATASET_ID or resource_ref.get("repo_type") != "dataset":
        fail("required_datasets.resource_ref 未正确声明数据集 repo_id/repo_type")

    command_refs = collect_command_refs(manifest)
    unresolved = sorted({ref for ref in command_refs if not resolve_command_ref(manifest, ref)})
    if unresolved:
        fail(f"Manifest command_refs 无法解析: {', '.join(unresolved)}")

    for stage, commands in manifest.get("commands", {}).items():
        if not isinstance(commands, list):
            fail(f"commands.{stage} 必须是列表")
        names = [item.get("name") for item in commands if isinstance(item, dict)]
        duplicates = [name for name, count in Counter(names).items() if name and count > 1]
        if duplicates:
            fail(f"commands.{stage} 存在重复 name: {', '.join(duplicates)}")

    known_dataset_ids = {item.get("id") for item in required_datasets + relations.get("optional_datasets", [])}
    for scenario in manifest["run_matrix"].get("scenarios", []):
        for dataset in scenario.get("required_datasets", []):
            if dataset.get("id") not in known_dataset_ids:
                fail(f"run_matrix 场景 {scenario.get('name')} 引用未声明数据集: {dataset.get('id')}")

    yaml_text = open(path, "r", encoding="utf-8").read()
    bad_model_ids = re.findall(r"OneScience/FourCastNet(?=[\"'\s])", yaml_text)
    if bad_model_ids:
        fail("Manifest 中存在不带尾部斜杠的模型 ID")

    ok(f"Manifest 可解析且关键引用有效: {path}")


def main():
    parser = argparse.ArgumentParser(description="校验 FourCastNet ModelScope 标准模型包")
    parser.add_argument("--root", default=".", help="标准模型包根目录")
    args = parser.parse_args()

    readme_path = os.path.join(args.root, "README.md")
    manifest_path = os.path.join(args.root, "onescience_run_manifest.yaml")
    compat_manifest_path = os.path.join(args.root, "manifest.yaml")
    for path in (readme_path, manifest_path, compat_manifest_path):
        if not os.path.isfile(path):
            fail(f"文件不存在: {path}")

    validate_readme(readme_path)
    validate_manifest(manifest_path)
    validate_manifest(compat_manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        main_manifest = f.read()
    with open(compat_manifest_path, "r", encoding="utf-8") as f:
        compat_manifest = f.read()
    if main_manifest != compat_manifest:
        fail("onescience_run_manifest.yaml 与 manifest.yaml 内容不一致")
    ok("模型和数据集 ID 与上传目标 ID 完全一致")
    ok("relations 与 run_matrix 数据集引用可解析")
    ok("兼容 Manifest 副本内容一致")


if __name__ == "__main__":
    main()
