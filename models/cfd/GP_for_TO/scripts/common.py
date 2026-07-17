import json
import os
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"
PROBLEMS = ("doublepipe", "diffuser", "rugby", "pipebend")


def ensure_onescience_path(explicit_src=None):
    candidates = []
    if explicit_src:
        candidates.append(Path(explicit_src).expanduser())
    if os.environ.get("ONESCIENCE_SRC"):
        candidates.append(Path(os.environ["ONESCIENCE_SRC"]).expanduser())
    for parent in (PROJECT_ROOT, *PROJECT_ROOT.parents):
        candidates.append(parent / "refactor" / "onescience" / "src")
        candidates.append(parent / "onescience" / "src")

    for candidate in candidates:
        if (candidate / "onescience").is_dir():
            path = str(candidate.resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
            return candidate.resolve()
    return None


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["root"]


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_device(section):
    requested = section.get("device", "auto")
    gpu = int(section.get("gpu", 0))
    if requested == "auto":
        return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {requested}, but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def dtype_from_config(name):
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def set_nested(cfg, section, key, value):
    if value is not None:
        cfg[section][key] = value


def build_models(cfg, device, *, n_col_domain=None, n_train_per_bc=None, problem=None):
    ensure_onescience_path(cfg.get("runtime", {}).get("onescience_src"))
    from model import GPPLUS
    from onescience.utils.GP_TO import get_data_fluid, set_seed

    problem = problem or cfg["problem"]
    n_col_domain = int(n_col_domain or cfg["data"]["n_col_domain"])
    n_train_per_bc = int(n_train_per_bc or cfg["data"]["n_train_per_bc"])
    dtype = dtype_from_config(cfg["model"].get("dtype", "float32"))

    set_seed(int(cfg["seed"]))
    x_col, x_train, sol_train = get_data_fluid(
        problem=problem,
        N_col_domain=n_col_domain,
        N_train=n_train_per_bc,
    )
    collocation_x = x_col.to(device=device, dtype=dtype).clone().requires_grad_(True)

    models = []
    for i, name in enumerate(cfg["output_names"]):
        model = GPPLUS(
            train_x=x_train[i].type(dtype),
            train_y=sol_train[i].type(dtype),
            collocation_x=collocation_x,
            basis=cfg["model"]["mean_function"],
            NN_layers_base=cfg["model"]["nn_layers_base"],
            name_output=name,
            device=device,
            dtype=dtype,
        ).to(device=device, dtype=dtype)
        models.append(model)

    return models, {
        "problem": problem,
        "n_col_domain": n_col_domain,
        "n_train_per_bc": n_train_per_bc,
        "x_col_shape": tuple(x_col.shape),
        "x_train_shapes": [tuple(x.shape) for x in x_train],
        "sol_train_shapes": [tuple(y.shape) for y in sol_train],
    }


def save_checkpoint(path, model_list, cfg, metadata, loss_history):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dicts": [model.state_dict() for model in model_list],
            "config": cfg,
            "metadata": metadata,
            "loss_history": loss_history,
        },
        path,
    )
    return path


def load_checkpoint(path, model_list, device):
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run scripts/train.py first.")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    state_dicts = checkpoint.get("model_state_dicts")
    if state_dicts is None:
        raise KeyError(f"Checkpoint does not contain model_state_dicts: {path}")
    for model, state_dict in zip(model_list, state_dicts):
        model.load_state_dict(state_dict)
    return checkpoint


def dump_json(data, path):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def tensor_to_numpy_dict(fields):
    return {key: value.detach().cpu().numpy() for key, value in fields.items()}
