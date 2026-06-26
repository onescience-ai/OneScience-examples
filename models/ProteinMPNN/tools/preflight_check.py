#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required to parse manifest YAML: {exc}")

MODEL_ID = "OneScience/ProteinMPNN"
DATASET_ID = "OneScience/proteinmpnn"
REQUIRED_TOP_KEYS = [
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
TEXT_FILES = [
    "README.md",
    "manifest.yaml",
    "onescience_run_manifest.yaml",
    "onescience_relations.yaml",
]
README_SECTIONS = [
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
]
REQUIRED_FILES = [
    "README.md",
    "manifest.yaml",
    "onescience_run_manifest.yaml",
    "onescience_relations.yaml",
    "protein_mpnn_run.py",
    "helper_scripts/parse_multiple_chains.py",
    "train/training.py",
    "tools/preflight_check.py",
    "tools/preflight_proteinmpnn.py",
    "tools/run_minimal_inference.sh",
    "inputs/PDB_monomers/pdbs/5L33.pdb",
    "vanilla_model_weights/v_48_002.pt",
    "vanilla_model_weights/v_48_010.pt",
    "vanilla_model_weights/v_48_020.pt",
    "vanilla_model_weights/v_48_030.pt",
    "soluble_model_weights/v_48_002.pt",
    "soluble_model_weights/v_48_010.pt",
    "soluble_model_weights/v_48_020.pt",
    "soluble_model_weights/v_48_030.pt",
    "soluble_model_weights/excluded_PDBs.csv",
    "ca_model_weights/v_48_002.pt",
    "ca_model_weights/v_48_010.pt",
    "ca_model_weights/v_48_020.pt",
]
BAD_PATTERNS = [
    re.compile(r"\?{4,}"),
    re.compile("\ufffd"),
    re.compile(r"(涓|绋|璧|妯|鐨|鍙|骞|棰|缁|浠|鏁|熴|銆|锛|鈥)"),
]


def read_text_checked(path: Path) -> tuple[str, int]:
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"UTF-8 decode failed: {path}: {exc}")
    for pattern in BAD_PATTERNS:
        if pattern.search(text):
            raise SystemExit(f"encoding corruption pattern found in {path}: {pattern.pattern}")
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return text, cjk_count


def load_yaml(path: Path):
    text, _ = read_text_checked(path)
    try:
        return yaml.safe_load(text)
    except Exception as exc:
        raise SystemExit(f"YAML parse failed: {path}: {exc}")


def walk_values(obj, key_name=None):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from walk_values(value, key)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_values(value, key_name)


def collect_command_names(commands: dict) -> set[str]:
    names = set()
    for entries in commands.values():
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and entry.get("name"):
                    names.add(entry["name"])
        elif isinstance(entries, dict):
            for value in entries.values():
                if isinstance(value, dict) and value.get("name"):
                    names.add(value["name"])
    return names


def collect_command_refs(obj) -> list[str]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()

    if not root.exists():
        raise SystemExit(f"repo root not found: {root}")

    cjk_counts = {}
    for rel in TEXT_FILES:
        text, cjk = read_text_checked(root / rel)
        cjk_counts[rel] = cjk
        if rel == "README.md":
            first_line = text.splitlines()[0] if text.splitlines() else ""
            if first_line != "# ProteinMPNN":
                raise SystemExit(f"README first line must be # ProteinMPNN, got: {first_line!r}")
            missing_sections = [section for section in README_SECTIONS if section not in text]
            if missing_sections:
                raise SystemExit(f"README missing sections: {missing_sections}")
            if cjk < 200:
                raise SystemExit(f"README CJK count too low: {cjk}")

    manifest = load_yaml(root / "manifest.yaml")
    run_manifest = load_yaml(root / "onescience_run_manifest.yaml")
    relations = load_yaml(root / "onescience_relations.yaml")

    missing = [key for key in REQUIRED_TOP_KEYS if key not in manifest]
    if missing:
        raise SystemExit(f"missing manifest keys: {missing}")

    if run_manifest != manifest:
        raise SystemExit("onescience_run_manifest.yaml must match manifest.yaml")

    if manifest["resource"]["id"] != MODEL_ID:
        raise SystemExit(f"resource.id mismatch: {manifest['resource']['id']}")
    primary = manifest["platform_resource"]["primary"]
    if primary.get("repo_id") != MODEL_ID or primary.get("repo_type") != "model":
        raise SystemExit(f"platform_resource.primary mismatch: {primary}")
    if primary.get("revision") != "master":
        raise SystemExit(f"platform_resource.primary.revision must be master: {primary.get('revision')}")
    if manifest["runtime_package"].get("kind") == "artifact_only":
        raise SystemExit("runtime_package.kind must not be artifact_only")

    for key, value in walk_values(manifest):
        if key == "repo_id":
            if value not in {MODEL_ID, DATASET_ID}:
                raise SystemExit(f"unexpected repo_id value: {value}")
        if isinstance(value, str):
            if "Onescience/" in value or "onescience/" in value:
                raise SystemExit(f"bad namespace casing found: {value}")
            if "modelscope download --model ProteinMPNN" in value:
                raise SystemExit("bare model download command found")

    required_datasets = manifest["relations"].get("required_datasets", [])
    if not required_datasets:
        raise SystemExit("relations.required_datasets is empty")
    dataset_refs = [item.get("resource_ref", {}).get("repo_id") for item in required_datasets]
    if DATASET_ID not in dataset_refs:
        raise SystemExit(f"required dataset relation missing {DATASET_ID}: {dataset_refs}")

    rel_refs = relations.get("relations", {}).get("required_datasets", [])
    if not rel_refs or rel_refs[0].get("resource_ref", {}).get("repo_id") != DATASET_ID:
        raise SystemExit("onescience_relations.yaml missing dataset resource_ref")

    command_names = collect_command_names(manifest["commands"])
    refs = collect_command_refs(manifest.get("files", {})) + collect_command_refs(manifest.get("run_matrix", {}))
    unresolved = sorted({ref for ref in refs if ref not in command_names})
    if unresolved:
        raise SystemExit(f"unresolved command_ref values: {unresolved}; commands={sorted(command_names)}")

    missing_files = [rel for rel in REQUIRED_FILES if not (root / rel).exists()]
    if missing_files:
        raise SystemExit(f"missing required files: {missing_files}")

    print("MODEL_PREFLIGHT_OK")
    print(f"repo_root={root}")
    print(f"resource_id={manifest['resource']['id']}")
    print(f"dataset_repo_id={DATASET_ID}")
    print(f"command_refs_checked={len(refs)}")
    print(f"required_files_checked={len(REQUIRED_FILES)}")
    print("cjk_counts=" + ",".join(f"{k}:{v}" for k, v in sorted(cjk_counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
