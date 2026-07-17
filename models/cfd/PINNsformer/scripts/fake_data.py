from __future__ import annotations

import argparse

import numpy as np

from common import exact_reaction_solution, load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake PINNsformer reaction data.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    x = np.linspace(*data_cfg["x_range"], int(data_cfg["x_num"]), dtype=np.float32)
    t = np.linspace(*data_cfg["t_range"], int(data_cfg["t_num"]), dtype=np.float32)
    x_mesh, t_mesh = np.meshgrid(x, t)
    u = exact_reaction_solution(x_mesh, t_mesh, cfg).astype(np.float32)

    output_path = project_path(cfg["paths"]["fake_data"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, x=x, t=t, u=u)

    print(f"fake data saved to {output_path}")
    print(f"x={x.shape}, t={t.shape}, u={u.shape}")


if __name__ == "__main__":
    main()
