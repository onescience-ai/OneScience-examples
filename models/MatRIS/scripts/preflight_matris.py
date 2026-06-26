#!/usr/bin/env python3
"""Preflight check for the standardized MatRIS runtime package."""

from __future__ import annotations

import argparse
import bz2
import csv
from pathlib import Path


REQUIRED_COLUMNS = [
    "mp_id",
    "nsites",
    "energy_pa",
    "volume_pa",
    "entropy",
    "heat_capacity",
    "free_energy",
    "max_freq",
    "avg_freq",
    "stable",
]


def check_csv(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[FAIL] 数据文件不存在: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise SystemExit(f"[FAIL] CSV 表头不匹配: {path}: {reader.fieldnames}")
        rows = 0
        for row in reader:
            rows += 1
            if not row["mp_id"]:
                raise SystemExit(f"[FAIL] mp_id 为空: {path} row {rows}")
            int(row["nsites"])
            for key in ("volume_pa", "entropy", "heat_capacity", "free_energy", "max_freq", "avg_freq"):
                if row[key] != "":
                    float(row[key])
            if row["energy_pa"] != "":
                float(row["energy_pa"])
            if row["stable"] not in ("True", "False"):
                raise SystemExit(f"[FAIL] stable 字段异常: {path} row {rows}")
    if rows == 0:
        raise SystemExit(f"[FAIL] CSV 无数据行: {path}")
    print(f"[OK] CSV 可读: {path} rows={rows}")


def resolve_pbe_csvs(data_dir: Path) -> tuple[Path, Path]:
    standard = (data_dir / "pbe.csv", data_dir / "pbesol.csv")
    if standard[0].is_file() and standard[1].is_file():
        return standard

    legacy = (data_dir / "pbe_phonon_ref.csv", data_dir / "pbesol_phonon_ref.csv")
    if legacy[0].is_file() and legacy[1].is_file():
        print("[OK] 使用旧版 data/matris/*_phonon_ref.csv 兼容布局")
        return legacy

    raise SystemExit(
        "[FAIL] 数据文件不存在: 需要 data/pbe.csv 和 data/pbesol.csv，"
        "或旧版 data/matris/pbe_phonon_ref.csv 和 data/matris/pbesol_phonon_ref.csv"
    )


def check_pbe_yaml_bz2_dir(data_dir: Path, sample_count: int) -> None:
    pbe_dir = data_dir / "pbe"
    if not pbe_dir.exists():
        print(f"[OK] 未发现 {pbe_dir}，仅检查 CSV；如已下载 OneScience/pbe 完整数据集应包含该目录")
        return
    files = sorted(pbe_dir.glob("*.yaml.bz2"))
    if len(files) != 9958:
        raise SystemExit(f"[FAIL] yaml.bz2 文件数量异常: {pbe_dir} count={len(files)}, expected=9958")
    for path in files[:sample_count]:
        with bz2.open(path, "rt", encoding="utf-8") as handle:
            text = handle.read(4096)
        if "displacements:" not in text:
            raise SystemExit(f"[FAIL] YAML.BZ2 内容异常: {path}")
    print(f"[OK] YAML.BZ2 目录可读: {pbe_dir} files={len(files)} sampled={min(sample_count, len(files))}")


def check_matris_import_and_forward() -> None:
    import torch
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure

    from onescience.datapipes.materials.matris import GraphConverter
    from onescience.models.matris import MatRIS
    from onescience.utils.matris import MatRISCalculator, StructOptimizer  # noqa: F401

    model = MatRIS(
        num_layers=2,
        node_feat_dim=64,
        edge_feat_dim=64,
        three_body_feat_dim=64,
        num_radial=5,
        num_angular=5,
        pairwise_cutoff=5.0,
        three_body_cutoff=3.0,
        reference_energy=None,
    )
    model.eval()

    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    graph = GraphConverter(atom_graph_cutoff=5.0, line_graph_cutoff=3.0)(structure)
    with torch.no_grad():
        out = model([graph], task="efsm")
    for key in ("e", "f", "s", "m"):
        if key not in out:
            raise SystemExit(f"[FAIL] MatRIS 前向输出缺少键: {key}")
    print("[OK] MatRIS 模块导入和 CPU 前向传播通过")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-bz2", type=int, default=4)
    parser.add_argument(
        "--skip-model-forward",
        action="store_true",
        help="Only validate packaged CSV files; skip torch/pymatgen/OneScience imports.",
    )
    args = parser.parse_args()

    package_root = Path.cwd()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = package_root / data_dir

    pbe_csv, pbesol_csv = resolve_pbe_csvs(data_dir)
    check_csv(pbe_csv)
    check_csv(pbesol_csv)
    check_pbe_yaml_bz2_dir(data_dir, args.sample_bz2)
    if args.skip_model_forward:
        print("[OK] 已跳过 MatRIS 模块导入和 CPU 前向传播检查")
    else:
        check_matris_import_and_forward()
    print("[OK] MatRIS 标准运行包预检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
