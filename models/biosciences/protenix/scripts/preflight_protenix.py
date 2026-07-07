#!/usr/bin/env python3
import argparse
import hashlib
import os
from pathlib import Path


MODEL_SHA256 = "9ea20b0aba42f2256711da1d0cd081510a4b291e64375bff6b70ced70b87a5f1"
MODEL_SIZE = 1474265486

REQUIRED_MODEL_FILES = [
    "checkpoints/model_v0.5.0.pt",
    "runner/inference_unified.py",
    "runner/train.py",
    "configs/inference_config.yaml",
    "infer_datasets/7r6r.json",
    "infer_datasets/7r6r/msa/1/pairing.a3m",
    "infer_datasets/7r6r/msa/1/non_pairing.a3m",
]

REQUIRED_DATA_FILES = [
    "components.v20240608.cif",
    "components.v20240608.cif.rdkit_mol.pkl",
    "seq_to_pdb_index.json",
    "indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz",
    "indices/recentPDB_low_homology_maxtoken1536.csv",
    "indices/recentPDB_low_homology_maxtoken1024_sample384_pdb_id.txt",
    "indices/posebusters_indices_mainchain_interface.csv",
    "mmcif_msa",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Protenix 标准模型包预检")
    parser.add_argument("--model-root", default=".", help="模型标准仓库根目录")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT_DIR", "../bio_protenix_dataset"),
        help="Protenix 数据集根目录，默认读取 DATA_ROOT_DIR 或 ../bio_protenix_dataset",
    )
    parser.add_argument("--full-checksum", action="store_true", help="计算 1.4GB 权重 SHA256")
    args = parser.parse_args()

    model_root = Path(args.model_root).resolve()
    data_root = Path(args.data_root).resolve()
    errors: list[str] = []

    for rel in REQUIRED_MODEL_FILES:
        path = model_root / rel
        if not path.exists():
            errors.append(f"缺少模型包文件: {path}")

    checkpoint = model_root / "checkpoints/model_v0.5.0.pt"
    if checkpoint.exists():
        size = checkpoint.stat().st_size
        if size != MODEL_SIZE:
            errors.append(f"权重大小不匹配: {checkpoint} size={size} expected={MODEL_SIZE}")
        if args.full_checksum:
            digest = sha256(checkpoint)
            if digest != MODEL_SHA256:
                errors.append(f"权重 SHA256 不匹配: {checkpoint} sha256={digest}")

    config = model_root / "configs/inference_config.yaml"
    if config.exists():
        text = config.read_text(encoding="utf-8")
        required_fragments = [
            "load_checkpoint_path: \"./checkpoints/model_v0.5.0.pt\"",
            "input_json_path: \"./infer_datasets/7r6r.json\"",
            "ccd_components_file: \"${DATA_ROOT_DIR}/components.v20240608.cif\"",
            "ccd_components_rdkit_mol_file: \"${DATA_ROOT_DIR}/components.v20240608.cif.rdkit_mol.pkl\"",
            "pdb_mmseqs_dir: \"${DATA_ROOT_DIR}/mmcif_msa\"",
        ]
        for fragment in required_fragments:
            if fragment not in text:
                errors.append(f"推理配置缺少片段: {fragment}")

    for rel in REQUIRED_DATA_FILES:
        path = data_root / rel
        if not path.exists():
            errors.append(f"缺少数据集文件或目录: {path}")

    msa_root = data_root / "mmcif_msa"
    if msa_root.exists():
        found = False
        for _dirpath, _dirnames, filenames in os.walk(msa_root):
            if any(name.endswith(".a3m") for name in filenames):
                found = True
                break
        if not found:
            errors.append(f"mmcif_msa 下未发现 A3M 文件: {msa_root}")

    print(f"model_root={model_root}")
    print(f"data_root={data_root}")
    if errors:
        print("模型预检失败:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("模型预检通过: 配置、权重、示例输入和 Protenix 数据集路径匹配。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
