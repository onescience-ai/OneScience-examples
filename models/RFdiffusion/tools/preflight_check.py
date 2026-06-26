#!/usr/bin/env python3
import argparse
import hashlib
import re
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required to parse manifest.yaml: {exc}")

EXPECTED_REPO_ID = "OneScience/RFdiffusion"
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
REQUIRED_FILES = [
    "README.md",
    "manifest.yaml",
    "onescience_relations.yaml",
    "scripts/run_inference.py",
    "config/inference/base.yaml",
    "config/inference/symmetry.yaml",
    "examples/design_unconditional.sh",
    "examples/input_pdbs/1YCR.pdb",
    "models/Base_ckpt.pt",
    "models/Complex_base_ckpt.pt",
    "models/Complex_Fold_base_ckpt.pt",
    "models/InpaintSeq_ckpt.pt",
    "models/InpaintSeq_Fold_ckpt.pt",
    "models/ActiveSite_ckpt.pt",
    "models/Base_epoch8_ckpt.pt",
    "models/Complex_beta_ckpt.pt",
    "models/RF_structure_prediction_weights.pt",
]


def read_utf8(path: Path) -> str:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"UTF8_DECODE_FAIL {path}: {exc}")
    bad = []
    if "\ufffd" in text:
        bad.append("U+FFFD")
    if "????" in text:
        bad.append("four_question_marks")
    if re.search(r"(?:Ã|Â|Ð|Ñ|锛|涓|绠|鏂)", text):
        bad.append("possible_mojibake")
    if bad:
        raise SystemExit(f"ENCODING_CHECK_FAIL {path}: {','.join(bad)}")
    return text


def cjk_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def collect_repo_ids(obj):
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "repo_id":
                found.append(value)
            found.extend(collect_repo_ids(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(collect_repo_ids(item))
    return found


def collect_command_refs(obj):
    refs = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "command_ref":
                refs.append(value)
            else:
                refs.extend(collect_command_refs(value))
    elif isinstance(obj, list):
        for item in obj:
            refs.extend(collect_command_refs(item))
    return refs


def collect_command_names(commands):
    names = set()
    if isinstance(commands, dict):
        for value in commands.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("name"):
                        names.add(item["name"])
            elif isinstance(value, dict) and value.get("name"):
                names.add(value["name"])
    return names


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()

    readme = read_utf8(root / "README.md")
    manifest_text = read_utf8(root / "manifest.yaml")
    relations_text = read_utf8(root / "onescience_relations.yaml")
    manifest = yaml.safe_load(manifest_text)
    yaml.safe_load(relations_text)

    missing_keys = [key for key in REQUIRED_TOP_KEYS if key not in manifest]
    if missing_keys:
        raise SystemExit(f"MISSING_MANIFEST_KEYS {missing_keys}")

    if manifest["resource"]["id"] != EXPECTED_REPO_ID:
        raise SystemExit(f"BAD_RESOURCE_ID {manifest['resource']['id']}")
    primary = manifest["platform_resource"]["primary"]
    if primary.get("repo_id") != EXPECTED_REPO_ID or primary.get("repo_type") != "model":
        raise SystemExit(f"BAD_PLATFORM_RESOURCE {primary}")

    for repo_id in collect_repo_ids(manifest):
        if repo_id != EXPECTED_REPO_ID:
            raise SystemExit(f"BAD_REPO_ID {repo_id}")
    if "Onescience/" in readme or "onescience/" in readme:
        raise SystemExit("BAD_NAMESPACE_CASE_IN_README")
    bad_download_patterns = [
        "modelscope download --model RFdiffusion",
        "modelscope download RFdiffusion",
        "Onescience/RFdiffusion",
        "onescience/RFdiffusion",
    ]
    for pattern in bad_download_patterns:
        if pattern in readme or pattern in manifest_text or pattern in relations_text:
            raise SystemExit(f"BAD_REPO_REFERENCE {pattern}")

    readme_cjk = cjk_count(readme)
    manifest_cjk = cjk_count(manifest_text)
    if readme_cjk < 200 or manifest_cjk < 50:
        raise SystemExit(f"CJK_COUNT_TOO_LOW README={readme_cjk} MANIFEST={manifest_cjk}")

    missing_files = [rel for rel in REQUIRED_FILES if not (root / rel).exists()]
    if missing_files:
        raise SystemExit(f"MISSING_REQUIRED_FILES {missing_files}")

    command_names = collect_command_names(manifest.get("commands", {}))
    command_refs = collect_command_refs(manifest)
    missing_refs = sorted({ref for ref in command_refs if ref not in command_names})
    if missing_refs:
        raise SystemExit(f"MISSING_COMMAND_REFS {missing_refs}")

    if manifest["relations"].get("required_datasets") != []:
        raise SystemExit("REQUIRED_DATASETS_MUST_BE_EMPTY_FOR_THIS_PACKAGE")

    weights = sorted((root / "models").glob("*.pt"))
    total_weight_bytes = sum(p.stat().st_size for p in weights)
    if len(weights) < 9 or total_weight_bytes < 4_000_000_000:
        raise SystemExit(f"WEIGHTS_INCOMPLETE count={len(weights)} bytes={total_weight_bytes}")

    if args.write_checksums:
        checksum_lines = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
        (root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print("MODEL_PREFLIGHT_OK")
    print(f"manifest_resource={manifest['resource']['id']}")
    print(f"readme_cjk={readme_cjk}")
    print(f"manifest_cjk={manifest_cjk}")
    print(f"weights_checked={len(weights)}")
    print(f"weight_bytes={total_weight_bytes}")


if __name__ == "__main__":
    main()
