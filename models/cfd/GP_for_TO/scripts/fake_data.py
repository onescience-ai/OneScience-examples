import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import PROBLEMS, ensure_onescience_path, load_config, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GP_for_TO runtime sample tensors.")
    parser.add_argument("--problem", choices=PROBLEMS, default=None)
    parser.add_argument("--n-col-domain", type=int, default=None)
    parser.add_argument("--n-train-per-bc", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    ensure_onescience_path(cfg.get("runtime", {}).get("onescience_src"))
    from onescience.utils.GP_TO import get_data_fluid, set_seed

    problem = args.problem or cfg["fake_data"]["problem"]
    n_col_domain = args.n_col_domain or cfg["fake_data"]["n_col_domain"]
    n_train_per_bc = args.n_train_per_bc or cfg["fake_data"]["n_train_per_bc"]
    output_dir = resolve_path(args.output_dir or cfg["fake_data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["seed"]))
    x_col, x_train, sol_train = get_data_fluid(
        problem=problem,
        N_col_domain=n_col_domain,
        N_train=n_train_per_bc,
    )

    npz_path = output_dir / f"{problem}_samples.npz"
    arrays = {"x_col": x_col.cpu().numpy()}
    for i, name in enumerate(cfg["output_names"]):
        arrays[f"x_train_{name}"] = x_train[i].cpu().numpy()
        arrays[f"target_{name}"] = sol_train[i].cpu().numpy()
    np.savez(npz_path, **arrays)

    metadata = {
        "problem": problem,
        "n_col_domain_requested": int(n_col_domain),
        "n_train_per_bc": int(n_train_per_bc),
        "x_col_shape": list(x_col.shape),
        "x_train_shapes": [list(x.shape) for x in x_train],
        "target_shapes": [list(y.shape) for y in sol_train],
    }
    metadata_path = output_dir / f"{problem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Fake GP_for_TO tensors written to {npz_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
