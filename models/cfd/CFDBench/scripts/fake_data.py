import json
import shutil
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from onescience.utils.YParams import YParams


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_case(case_dir, case_id, rows, cols, frames, rng):
    case_dir.mkdir(parents=True, exist_ok=True)
    vel_in = 1.0 + 0.05 * case_id
    params = {
        "vel_in": vel_in,
        "density": 1.0 + 0.1 * (case_id % 3),
        "viscosity": 0.001 + 0.0001 * (case_id % 5),
        "height": 1.0,
        "width": 1.0,
    }

    y = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    x = np.linspace(0.0, 1.0, cols, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    u = np.empty((frames, rows, cols), dtype=np.float32)
    v = np.empty_like(u)

    phase = 0.2 * case_id
    for t in range(frames):
        amp = vel_in * (1.0 + 0.03 * t)
        u[t] = amp * np.sin(np.pi * yy) * (1.0 - 0.3 * xx) + 0.01 * rng.standard_normal((rows, cols))
        v[t] = 0.2 * amp * np.cos(np.pi * xx + phase) * yy + 0.01 * rng.standard_normal((rows, cols))

    (case_dir / "case.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    np.save(case_dir / "u.npy", u)
    np.save(case_dir / "v.npy", v)


def main():
    cfg = YParams(str(PROJECT_ROOT / "conf" / "config.yaml"), "root")
    data_dir = resolve_path(cfg.datapipe.source.data_dir)
    fake_cfg = cfg.fake_data
    data_cfg = cfg.datapipe.data

    if data_dir.exists():
        shutil.rmtree(data_dir)

    rng = np.random.default_rng(fake_cfg.seed)
    problem_dir = data_dir / "tube"
    case_id = 0
    for subset in ("prop", "bc", "geo"):
        for subset_idx in range(fake_cfg.num_cases_per_subset):
            make_case(
                problem_dir / subset / f"case{case_id}",
                case_id=case_id + subset_idx,
                rows=data_cfg.num_rows,
                cols=data_cfg.num_cols,
                frames=fake_cfg.num_frames,
                rng=rng,
            )
            case_id += 1

    total_cases = fake_cfg.num_cases_per_subset * 3
    print(f"Fake CFDBench data written to {data_dir}")
    print(f"cases={total_cases}, frames={fake_cfg.num_frames}, grid=({data_cfg.num_rows}, {data_cfg.num_cols})")


if __name__ == "__main__":
    main()
