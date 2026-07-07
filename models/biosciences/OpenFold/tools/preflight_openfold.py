#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:  # pragma: no cover
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


REPO_ID = "OneScience/OpenFold"
REQUIRED_FILES = [
    "README.md",
    "manifest.yaml",
    "onescience_run_manifest.yaml",
    "onescience_relations.yaml",
    "checksums.sha256",
    "params/finetuning_ptm_2.pt",
    "run_pretrained_openfold.py",
    "train_openfold.py",
    "monomer/inference.sh",
    "monomer/fasta_dir/6kwc.fasta",
    "tools/preflight_openfold.py",
    "tools/preflight_check.py",
]
TEXT_FILES = [
    "README.md",
    "manifest.yaml",
    "onescience_run_manifest.yaml",
    "onescience_relations.yaml",
]
YAML_FILES = [
    "manifest.yaml",
    "onescience_run_manifest.yaml",
    "onescience_relations.yaml",
]
FORBIDDEN_MARKERS = ["????", "\ufffd", "璇", "鐨", "銆"]


def fail(message: str) -> int:
    print(f"OpenFold preflight FAILED: {message}")
    return 1


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def nested_values(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from nested_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from nested_values(value)
    else:
        yield obj


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preflight check for OneScience/OpenFold package.")
    parser.add_argument("--repo-root", "--root", default=".", dest="root", help="OpenFold package root")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    missing = [rel for rel in REQUIRED_FILES if not (root / rel).exists()]
    if missing:
        return fail("missing required file(s): " + ", ".join(missing))

    if yaml is None:
        return fail(f"PyYAML import failed: {YAML_IMPORT_ERROR}")

    texts = {}
    for rel in TEXT_FILES:
        try:
            text = read_utf8(root / rel)
        except UnicodeDecodeError as exc:
            return fail(f"{rel} is not valid UTF-8: {exc}")
        texts[rel] = text
        if not any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return fail(f"{rel} has no CJK content")
        for marker in FORBIDDEN_MARKERS:
            if marker in text:
                return fail(f"{rel} contains forbidden mojibake marker {marker!r}")

    if texts["README.md"].splitlines()[0].strip() != "# OpenFold":
        return fail("README.md first line must be exactly '# OpenFold'")

    parsed = {}
    for rel in YAML_FILES:
        try:
            parsed[rel] = yaml.safe_load(texts[rel])
        except Exception as exc:
            return fail(f"{rel} YAML parse failed: {exc}")

    manifest = parsed["manifest.yaml"]
    run_manifest = parsed["onescience_run_manifest.yaml"]
    relations_doc = parsed["onescience_relations.yaml"]

    if manifest.get("resource", {}).get("id") != REPO_ID:
        return fail("manifest resource.id mismatch")
    if manifest.get("resource", {}).get("name") != "OpenFold":
        return fail("manifest resource.name mismatch")
    if manifest.get("resource", {}).get("chinese_name") != "OpenFold":
        return fail("manifest resource.chinese_name mismatch")
    primary = manifest.get("platform_resource", {}).get("primary", {})
    if primary.get("repo_id") != REPO_ID:
        return fail("manifest platform_resource.primary.repo_id mismatch")
    if primary.get("revision") != "master":
        return fail("manifest platform_resource.primary.revision must be master")
    run_primary = run_manifest.get("platform_resource", {}).get("primary", {})
    if run_primary.get("repo_id") != REPO_ID:
        return fail("run manifest repo_id mismatch")
    if run_primary.get("revision") != "master":
        return fail("run manifest revision must be master")
    if relations_doc.get("resource", {}).get("id") != REPO_ID:
        return fail("relations resource.id mismatch")

    manifest_rel = manifest.get("relations", {})
    file_rel = relations_doc.get("relations", {})
    for label, rel_obj in (("manifest", manifest_rel), ("relations", file_rel)):
        if rel_obj.get("required_datasets") != []:
            return fail(f"{label} required_datasets must be empty")
        if rel_obj.get("optional_datasets") != []:
            return fail(f"{label} optional_datasets must be empty")

    combined_text = "\n".join(texts.values())
    if "OneScience/proteinmpnn" in combined_text:
        return fail("forbidden dataset relation OneScience/proteinmpnn found")
    for value in nested_values(parsed):
        if isinstance(value, str) and "OneScience/proteinmpnn" in value:
            return fail("forbidden dataset relation OneScience/proteinmpnn found in YAML")

    checkpoint = root / "params/finetuning_ptm_2.pt"
    size_mb = checkpoint.stat().st_size / 1024 / 1024
    print(f"checkpoint: {checkpoint.relative_to(root)} {size_mb:.1f} MiB")
    print("OpenFold preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
