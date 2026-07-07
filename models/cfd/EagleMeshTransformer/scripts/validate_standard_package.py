#!/usr/bin/env python3
import sys
from pathlib import Path

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "model" / "cfd_EagleMeshTransformer"
DATASET_DIR = ROOT / "dataset" / "cfd_eagle"
MODEL_ID = "OneScience/EagleMeshTransformer"
DATASET_ID = "OneScience/eagle"


def fail(message):
    raise SystemExit(f"[ERROR] {message}")


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return YAML(typ="safe").load(f)


def command_names(manifest, stage):
    return {item["name"] for item in manifest["commands"].get(stage, [])}


def validate_commands(manifest):
    for scenario in manifest["run_matrix"]["scenarios"]:
        for ref in scenario.get("command_refs", []):
            parts = ref.split(".")
            if len(parts) != 3 or parts[0] != "commands":
                fail(f"invalid command_ref format: {ref}")
            if parts[2] not in command_names(manifest, parts[1]):
                fail(f"unresolved command_ref: {ref}")


def validate_readme(path, sections):
    text = path.read_text(encoding="utf-8")
    missing = [section for section in sections if f"## {section}" not in text]
    if missing:
        fail(f"{path} missing README sections: {missing}")


def validate_ids(manifest, expected_id, repo_type):
    if manifest["resource"]["id"] != expected_id:
        fail(f"resource.id mismatch: {manifest['resource']['id']} != {expected_id}")
    primary = manifest["platform_resource"]["primary"]
    if primary["repo_id"] != expected_id:
        fail(f"platform_resource.primary.repo_id mismatch: {primary['repo_id']} != {expected_id}")
    if primary["repo_type"] != repo_type:
        fail(f"repo_type mismatch: {primary['repo_type']} != {repo_type}")


def main():
    model_manifest = load_yaml(MODEL_DIR / "onescience_run_manifest.yaml")
    dataset_manifest = load_yaml(DATASET_DIR / "onescience_run_manifest.yaml")

    validate_ids(model_manifest, MODEL_ID, "model")
    validate_ids(dataset_manifest, DATASET_ID, "dataset")
    validate_commands(model_manifest)
    validate_commands(dataset_manifest)

    model_rel = model_manifest["relations"]["required_datasets"][0]["resource_ref"]
    data_rel = dataset_manifest["relations"]["compatible_models"][0]["resource_ref"]
    if model_rel["repo_id"] != DATASET_ID or data_rel["repo_id"] != MODEL_ID:
        fail("relations are not bidirectionally resolvable")

    validate_readme(
        MODEL_DIR / "README.md",
        [
            "OneScience 官方信息",
            "项目说明",
            "Resource Card",
            "文件说明",
            "Manifest",
            "模型 vs 数据集关系",
            "文件与下载",
            "环境安装",
            "运行流程",
            "预检与诊断",
            "输出说明",
            "限制与适用范围",
            "引用与许可证",
        ],
    )
    validate_readme(
        DATASET_DIR / "README.md",
        [
            "OneScience 官方信息",
            "项目说明",
            "Resource Card",
            "文件说明",
            "Manifest",
            "模型 vs 数据集关系",
            "文件与下载",
            "数据格式",
            "数据放置位置",
            "数据读取验证",
            "适用模型",
            "限制与许可证",
        ],
    )

    print("[OK] YAML parse, README sections, command_refs, IDs and bidirectional relations validated")


if __name__ == "__main__":
    main()
