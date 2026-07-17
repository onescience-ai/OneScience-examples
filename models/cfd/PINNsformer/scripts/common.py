from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else PROJECT_ROOT / "conf" / "config.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_runtime_dirs(cfg: dict[str, Any]) -> None:
    for key in ("fake_data", "prediction", "metrics", "loss", "figure"):
        ensure_parent(project_path(cfg["paths"][key]))
    ensure_parent(project_path(cfg["training"]["checkpoint"]))


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    from model import PINNsformer1D

    model_cfg = cfg["model"]
    return PINNsformer1D(
        d_out=int(model_cfg["d_out"]),
        d_hidden=int(model_cfg["d_hidden"]),
        d_model=int(model_cfg["d_model"]),
        N=int(model_cfg["num_layers"]),
        heads=int(model_cfg["heads"]),
    )


def initial_condition(x: np.ndarray | torch.Tensor, cfg: dict[str, Any]):
    initial = cfg["equation"]["initial"]
    center = initial["center"]
    sigma = initial["sigma"]
    if isinstance(x, torch.Tensor):
        return torch.exp(-((x - center) ** 2) / (2 * sigma**2))
    return np.exp(-((x - center) ** 2) / (2 * sigma**2))


def exact_reaction_solution(
    x: np.ndarray | torch.Tensor,
    t: np.ndarray | torch.Tensor,
    cfg: dict[str, Any],
):
    h = initial_condition(x, cfg)
    rate = cfg["equation"]["reaction_rate"]
    if isinstance(h, torch.Tensor):
        exp_term = torch.exp(rate * t)
    else:
        exp_term = np.exp(rate * t)
    return h * exp_term / (h * exp_term + 1 - h)


def relative_errors(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "relative_l1": float(np.sum(np.abs(target - pred)) / np.sum(np.abs(target))),
        "relative_l2": float(np.sqrt(np.sum((target - pred) ** 2) / np.sum(target**2))),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
