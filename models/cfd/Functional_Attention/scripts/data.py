"""Dataset checks and optional statistics generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from funcattn_airfrans.data.airfrans import (  # noqa: E402
    compute_airfrans_field_stats,
    load_airfrans_field_case,
)
from funcattn_airfrans.utils.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--compute-stats", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = cfg["data"]
    data_root = resolve_path(ROOT, data_cfg["root"])
    cache_dir = resolve_path(ROOT, data_cfg["cache_dir"])
    stats_path = resolve_path(ROOT, data_cfg["stats_path"])
    manifest_path = data_root / "manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"data_root={data_root}")
    for key, value in manifest.items():
        print(f"split={key} count={len(value)}")

    sample_name = manifest[data_cfg["train_split"]][0]
    sample = load_airfrans_field_case(
        data_root,
        sample_name,
        max_points=int(data_cfg["max_points"]),
        cache_dir=cache_dir,
    )
    print(
        f"sample={sample.name} x={sample.features.shape} y={sample.target.shape} "
        f"surf_points={int(sample.surf.sum())}"
    )

    if args.compute_stats:
        stats = compute_airfrans_field_stats(
            data_root,
            data_cfg["train_split"],
            stats_path=stats_path,
            cache_dir=cache_dir,
        )
        print(f"stats_path={stats_path}")
        print(f"mean_in={stats['mean_in'].tolist()}")
        print(f"std_out={stats['std_out'].tolist()}")


if __name__ == "__main__":
    main()
