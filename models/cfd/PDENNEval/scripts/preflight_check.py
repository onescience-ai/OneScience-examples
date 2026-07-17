from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from common import DEFAULT_CONFIG, load_config, prepare_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight the PDENNEval standard package.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to conf/config.yaml")
    parser.add_argument("--data-dir", default=None, help="Override datapipe.source.data_dir")
    args = parser.parse_args()

    cfg = prepare_config(load_config(args.config), data_dir=args.data_dir)
    data_path = Path(cfg.datapipe.source.data_dir) / cfg.datapipe.source.file_name
    if not data_path.is_file():
        raise FileNotFoundError(f"missing HDF5 data file: {data_path}")

    with h5py.File(data_path, "r") as handle:
        for key in ("tensor", "nu", "x-coordinate", "y-coordinate"):
            if key not in handle:
                raise ValueError(f"{data_path.name} missing required key: {key}")
        if handle["tensor"].ndim != 4:
            raise ValueError("2D Darcy fake/default config expects tensor ndim=4")
        if handle["nu"].ndim != 3:
            raise ValueError("2D Darcy fake/default config expects nu ndim=3")

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".preflight_write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    print(f"[OK] data file: {data_path}")
    print(f"[OK] output directory writable: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
