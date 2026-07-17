"""Evaluation for the MARIO AirfRANS reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from .data import (
    case_uinf_angle,
    deterministic_indices,
    load_airfrans_case,
    load_stats,
    split_names,
)
from .model import GeometryEncoder, MarioDecoder
from .train import build_decoder, build_geometry_encoder, device_from_config, load_checkpoint, load_config


def resolve_checkpoint_path(out_dir: Path, stem: str, preference: str) -> Path:
    if preference == "best":
        best_path = out_dir / f"{stem}_best.pt"
        if best_path.exists():
            return best_path
    return out_dir / f"{stem}_last.pt"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra = _rank(np.asarray(a))
    rb = _rank(np.asarray(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def encode_case(
    encoder: GeometryEncoder,
    case,
    device: torch.device,
    *,
    points: int,
    inner_steps: int,
    inner_lr: float,
) -> torch.Tensor:
    idx = deterministic_indices(case.name, case.coords.shape[0], points, salt="eval-geometry")
    coords = torch.from_numpy(case.coords[idx][None].astype(np.float32)).to(device)
    sdf = torch.from_numpy(case.sdf[idx][None].astype(np.float32)).to(device)
    mask = torch.ones(1, len(idx), dtype=torch.float32, device=device)
    return encoder.encode(coords, sdf, mask, steps=inner_steps, inner_lr=inner_lr, create_graph=False).detach()


@torch.no_grad()
def predict_case(
    decoder: MarioDecoder,
    case,
    latent: torch.Tensor,
    stats: dict[str, np.ndarray],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    condition = case.freestream.astype(np.float32)
    if bool(cfg["training"].get("normalize_condition", True)):
        condition = (condition - stats["condition_mean"]) / (stats["condition_std"] + 1e-8)
    cond = torch.from_numpy(condition[None]).to(device)
    cond = torch.cat([latent, cond], dim=-1)
    chunk = int(cfg["evaluation"].get("chunk_points", 65536))
    preds: list[np.ndarray] = []
    local = case.decoder_input
    for start in range(0, local.shape[0], chunk):
        stop = min(start + chunk, local.shape[0])
        x = torch.from_numpy(local[start:stop][None].astype(np.float32)).to(device)
        out = decoder(x, cond).squeeze(0).detach().cpu().numpy()
        preds.append(out)
    pred_norm = np.concatenate(preds, axis=0)
    pred_phys = pred_norm * (stats["target_std"] + 1e-8) + stats["target_mean"]
    return pred_norm.astype(np.float32), pred_phys.astype(np.float32)


def compute_force_metrics(
    cfg: dict[str, Any],
    names: list[str],
    pred_phys_by_name: dict[str, np.ndarray],
    surf_by_name: dict[str, np.ndarray],
) -> dict[str, Any]:
    try:
        import pyvista as pv
        onescience_src = Path("/public/home/liuyushuang/code/onescience/src")
        if onescience_src.exists() and str(onescience_src) not in sys.path:
            sys.path.insert(0, str(onescience_src))
        from onescience.utils.transolver.metrics import Compute_coefficients
    except Exception as exc:
        return {"status": "skipped", "reason": repr(exc)}

    data_root = Path(cfg["data_root"])
    true_coefs = []
    pred_coefs = []
    for index, name in enumerate(names, start=1):
        internal = pv.read(data_root / name / f"{name}_internal.vtu")
        aerofoil = pv.read(data_root / name / f"{name}_aerofoil.vtp")
        uinf, angle = case_uinf_angle(name)
        surf = surf_by_name[name]
        true = np.asarray(Compute_coefficients([internal], [aerofoil], surf, uinf, angle)[0], dtype=np.float64)

        pred_values = pred_phys_by_name[name].copy()
        pred_values[surf, :2] = 0.0
        pred_values[surf, 3] = 0.0
        pred_internal = internal.copy()
        pred_internal.point_data["U"][:, :2] = pred_values[:, :2]
        pred_internal.point_data["p"] = pred_values[:, 2]
        pred_internal.point_data["nut"] = pred_values[:, 3]
        pred = np.asarray(Compute_coefficients([pred_internal], [aerofoil], surf, uinf, angle)[0], dtype=np.float64)
        true_coefs.append(true)
        pred_coefs.append(pred)
        if index == 1 or index % 25 == 0 or index == len(names):
            print(f"force_metrics cases={index}/{len(names)}", flush=True)

    true_arr = np.stack(true_coefs, axis=0)
    pred_arr = np.stack(pred_coefs, axis=0)
    rel = np.abs((pred_arr - true_arr) / np.maximum(np.abs(true_arr), 1e-12))
    return {
        "status": "ok",
        "cd_mre_percent": float(rel[:, 0].mean() * 100.0),
        "cl_mre_percent": float(rel[:, 1].mean() * 100.0),
        "rho_d": spearman(true_arr[:, 0], pred_arr[:, 0]),
        "rho_l": spearman(true_arr[:, 1], pred_arr[:, 1]),
        "true_coefficients": true_arr,
        "pred_coefficients": pred_arr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--with-force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_from_config(cfg["training"])
    print(f"device={device}", flush=True)
    stats_path = Path(cfg.get("stats_path") or Path(cfg["training"]["output_dir"]) / "stats.npz")
    stats = load_stats(stats_path)

    encoder = build_geometry_encoder(cfg).to(device)
    decoder = build_decoder(cfg).to(device)
    out_dir = Path(cfg["training"]["output_dir"])
    checkpoint_preference = str(cfg.get("evaluation", {}).get("checkpoint", "last"))
    geom_path = resolve_checkpoint_path(out_dir, "geometry", checkpoint_preference)
    dec_path = resolve_checkpoint_path(out_dir, "decoder", checkpoint_preference)
    geom_state = load_checkpoint(geom_path, device)
    dec_state = load_checkpoint(dec_path, device)
    if geom_state is None or dec_state is None:
        raise FileNotFoundError(f"{geom_path} and {dec_path} are required for evaluation")
    encoder.load_state_dict(geom_state["model"])
    decoder.load_state_dict(dec_state["model"])
    encoder.eval()
    decoder.eval()

    split = args.split or cfg["test_split"]
    names = split_names(cfg["data_root"], split, args.max_cases if args.max_cases is not None else cfg["evaluation"].get("max_cases"))
    inner_steps = int(cfg["geometry"].get("inner_steps", 3))
    inner_lr = float(cfg["geometry"].get("inner_lr", 0.1))
    geometry_points = int(cfg["evaluation"].get("geometry_points", cfg["training"]["points_per_case"]))

    channel_sse = np.zeros(4, dtype=np.float64)
    channel_count = 0
    surface_p_sse = 0.0
    surface_p_count = 0
    pred_phys_by_name: dict[str, np.ndarray] = {}
    surf_by_name: dict[str, np.ndarray] = {}

    for index, name in enumerate(names, start=1):
        case = load_airfrans_case(
            cfg["data_root"],
            name,
            cache_dir=cfg.get("cache_dir"),
            tau=float(cfg["model"].get("boundary_layer_tau", 0.02)),
        )
        latent = encode_case(
            encoder,
            case,
            device,
            points=geometry_points,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
        )
        pred_norm, pred_phys = predict_case(decoder, case, latent, stats, cfg, device)
        target_norm = (case.target - stats["target_mean"]) / (stats["target_std"] + 1e-8)
        err = (pred_norm - target_norm) ** 2
        channel_sse += err.sum(axis=0)
        channel_count += err.shape[0]
        if np.any(case.surf):
            surf_err = err[case.surf, 2]
            surface_p_sse += float(surf_err.sum())
            surface_p_count += int(surf_err.shape[0])
        if args.with_force:
            pred_phys_by_name[name] = pred_phys
            surf_by_name[name] = case.surf
        if index == 1 or index % 25 == 0 or index == len(names):
            print(f"eval split={split} cases={index}/{len(names)}", flush=True)

    mse = channel_sse / max(channel_count, 1)
    surface_p_mse = surface_p_sse / max(surface_p_count, 1)
    paper_scaled = {
        "ux_x1e_minus_2_table_value": float(mse[0] * 100.0),
        "uy_x1e_minus_2_table_value": float(mse[1] * 100.0),
        "p_x1e_minus_2_table_value": float(mse[2] * 100.0),
        "nut_x1e_minus_1_table_value": float(mse[3] * 10.0),
        "surface_p_x1e_minus_1_table_value": float(surface_p_mse * 10.0),
    }
    result: dict[str, Any] = {
        "split": split,
        "cases": len(names),
        "normalized_mse": {
            "ux": float(mse[0]),
            "uy": float(mse[1]),
            "p": float(mse[2]),
            "nut": float(mse[3]),
            "surface_p": float(surface_p_mse),
        },
        "paper_table_scaled_mse": paper_scaled,
        "checkpoints": {
            "geometry_path": str(geom_path),
            "decoder_path": str(dec_path),
            "geometry_epoch": int(geom_state.get("epoch", -1)),
            "decoder_epoch": int(dec_state.get("epoch", -1)),
        },
    }
    if args.with_force:
        result["force_metrics"] = compute_force_metrics(cfg, names, pred_phys_by_name, surf_by_name)

    out_dir = Path(cfg["training"]["output_dir"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_metrics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_json_default)
    print(f"metrics={out_path}", flush=True)
    print(json.dumps(result["normalized_mse"], indent=2), flush=True)


if __name__ == "__main__":
    main()
