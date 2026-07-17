"""Training entry for the MARIO AirfRANS reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    AirfransCaseDataset,
    collate_cases,
    compute_output_stats,
    load_stats,
    split_names,
)
from .model import GeometryEncoder, MarioDecoder, masked_mse


CHANNEL_NAMES = ("ux", "uy", "p", "nut")
PAPER_TABLE_TARGETS = {
    "ux": 0.152,
    "uy": 0.113,
    "p": 0.240,
    "nut": 0.096,
    "surface_p": 0.270,
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def device_from_config(cfg: dict[str, Any]) -> torch.device:
    name = cfg.get("device", "auto")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def ensure_stats(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    stats_path = Path(cfg["training"]["output_dir"]) / "stats.npz"
    configured = cfg.get("stats_path")
    if configured:
        stats_path = Path(configured)
    if stats_path.exists():
        print(f"loading_stats={stats_path}", flush=True)
        return load_stats(stats_path)
    print(f"computing_stats={stats_path}", flush=True)
    return compute_output_stats(
        cfg["data_root"],
        cfg["train_split"],
        stats_path=stats_path,
        cache_dir=cfg.get("cache_dir"),
        limit=cfg["training"].get("max_train_samples"),
        tau=float(cfg["model"].get("boundary_layer_tau", 0.02)),
    )


def split_train_validation_names(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    names = split_names(
        cfg["data_root"],
        cfg["train_split"],
        limit=cfg["training"].get("max_train_samples"),
    )
    val_cfg = cfg.get("validation", {}) or {}
    if not bool(val_cfg.get("enabled", False)):
        return names, []

    explicit_count = val_cfg.get("count")
    if explicit_count is not None:
        val_count = int(explicit_count)
    else:
        ratio = float(val_cfg.get("split_ratio", 0.0))
        val_count = int(round(len(names) * ratio))
        if ratio > 0.0:
            val_count = max(int(val_cfg.get("min_cases", 1)), val_count)
    val_count = min(max(val_count, 0), max(len(names) - 1, 0))
    if val_count == 0:
        return names, []

    rng = np.random.default_rng(int(val_cfg.get("seed", cfg.get("seed", 42))))
    val_indices = set(int(i) for i in rng.choice(len(names), size=val_count, replace=False))
    if bool(val_cfg.get("holdout", True)):
        train_names = [name for idx, name in enumerate(names) if idx not in val_indices]
    else:
        train_names = list(names)
    val_names = [name for idx, name in enumerate(names) if idx in val_indices]
    return train_names, val_names


def make_dataset(
    cfg: dict[str, Any],
    split: str,
    stats: dict[str, np.ndarray],
    *,
    deterministic: bool,
    names: list[str] | None = None,
    points_per_case: int | None = None,
) -> AirfransCaseDataset:
    if names is None:
        limit = cfg["training"].get("max_train_samples") if split == cfg["train_split"] else cfg["training"].get("max_test_samples")
    else:
        limit = None
    return AirfransCaseDataset(
        cfg["data_root"],
        split,
        cache_dir=cfg.get("cache_dir"),
        points_per_case=int(points_per_case if points_per_case is not None else cfg["training"]["points_per_case"]),
        limit=limit,
        stats=stats,
        normalize_condition=bool(cfg["training"].get("normalize_condition", True)),
        deterministic=deterministic,
        tau=float(cfg["model"].get("boundary_layer_tau", 0.02)),
        base_seed=int(cfg.get("seed", 42)),
        preload=bool(cfg["training"].get("preload_cases", False)),
        names=names,
    )


def make_loader(cfg: dict[str, Any], dataset: AirfransCaseDataset, *, shuffle: bool, batch_size: int | None = None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size if batch_size is not None else cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        collate_fn=collate_cases,
        pin_memory=torch.cuda.is_available(),
    )


def build_geometry_encoder(cfg: dict[str, Any]) -> GeometryEncoder:
    model_cfg = cfg["model"]
    return GeometryEncoder(
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        hidden_layers=int(model_cfg["encoder_layers"]),
        fourier_features=int(model_cfg["fourier_features"]),
        fourier_sigma=float(model_cfg["fourier_sigma"]),
    )


def build_decoder(cfg: dict[str, Any]) -> MarioDecoder:
    model_cfg = cfg["model"]
    return MarioDecoder(
        local_dim=int(model_cfg["decoder_local_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        condition_dim=int(model_cfg["latent_dim"]) + 2,
        hidden_dim=int(model_cfg["hidden_dim"]),
        hidden_layers=int(model_cfg["decoder_layers"]),
        fourier_features=int(model_cfg["fourier_features"]),
        fourier_sigma=float(model_cfg["fourier_sigma"]),
        hyper_depth=3,
        hyper_hidden_dim=int(model_cfg.get("hyper_hidden_dim", model_cfg["hidden_dim"])),
    )


def checkpoint_path(cfg: dict[str, Any], name: str) -> Path:
    return Path(cfg["training"]["output_dir"]) / name


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def train_geometry(cfg: dict[str, Any], stats: dict[str, np.ndarray], device: torch.device) -> GeometryEncoder:
    dataset = make_dataset(cfg, cfg["train_split"], stats, deterministic=False)
    loader = make_loader(cfg, dataset, shuffle=True)
    encoder = build_geometry_encoder(cfg).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=float(cfg["geometry"]["lr"]))
    path = checkpoint_path(cfg, "geometry_last.pt")
    start_epoch = 0
    if bool(cfg["geometry"].get("resume", True)):
        state = load_checkpoint(path, device)
        if state is not None:
            encoder.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            set_optimizer_lr(optimizer, float(cfg["geometry"]["lr"]))
            start_epoch = int(state.get("epoch", 0))
            print(f"resumed_geometry={path} start_epoch={start_epoch}", flush=True)
            print(f"geometry_lr={optimizer.param_groups[0]['lr']}", flush=True)

    epochs = int(cfg["geometry"]["epochs"])
    inner_steps = int(cfg["geometry"].get("inner_steps", 3))
    inner_lr = float(cfg["geometry"].get("inner_lr", 0.1))
    second_order = bool(cfg["geometry"].get("second_order", True))
    for epoch in range(start_epoch + 1, epochs + 1):
        dataset.set_epoch(epoch)
        encoder.train()
        running = 0.0
        for batch in loader:
            coords = batch["coords"].to(device, non_blocking=True)
            sdf = batch["sdf"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            latent = encoder.encode(coords, sdf, mask, steps=inner_steps, inner_lr=inner_lr, create_graph=second_order)
            pred = encoder.predict_sdf(coords, latent)
            loss = masked_mse(pred, sdf, mask)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
        avg = running / max(len(loader), 1)
        print(f"stage=geometry epoch={epoch} train_sdf_mse={avg:.8f}", flush=True)
        torch.save(
            {
                "model": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": cfg,
                "loss": avg,
            },
            path,
        )
    return encoder


@torch.no_grad()
def _empty_latent_map(names: list[str], latent_dim: int) -> dict[str, np.ndarray]:
    return {name: np.zeros(latent_dim, dtype=np.float32) for name in names}


def encode_split(
    cfg: dict[str, Any],
    stats: dict[str, np.ndarray],
    encoder: GeometryEncoder,
    split: str,
    device: torch.device,
    *,
    out_path: Path,
) -> dict[str, np.ndarray]:
    dataset = make_dataset(cfg, split, stats, deterministic=True)
    dataset.set_epoch(0)
    loader = make_loader(cfg, dataset, shuffle=False)
    encoder.eval()
    latents: dict[str, np.ndarray] = {}
    inner_steps = int(cfg["geometry"].get("inner_steps", 3))
    inner_lr = float(cfg["geometry"].get("inner_lr", 0.1))
    for batch in loader:
        coords = batch["coords"].to(device, non_blocking=True)
        sdf = batch["sdf"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        latent = encoder.encode(coords, sdf, mask, steps=inner_steps, inner_lr=inner_lr, create_graph=False)
        for name, value in zip(batch["name"], latent.detach().cpu().numpy()):
            latents[name] = value.astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **latents)
    print(f"saved_latents={out_path} count={len(latents)}", flush=True)
    return latents


def load_latents(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def batch_condition(batch: dict[str, Any], latent_map: dict[str, np.ndarray], device: torch.device) -> torch.Tensor:
    latent = torch.from_numpy(np.stack([latent_map[name] for name in batch["name"]], axis=0)).to(device)
    condition = batch["condition"].to(device, non_blocking=True)
    return torch.cat([latent, condition], dim=-1)


def _channel_weights(cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    raw = (cfg.get("loss", {}) or {}).get("channel_weights", [1.0, 1.0, 1.0, 1.0])
    if isinstance(raw, dict):
        values = [float(raw.get(name, 1.0)) for name in CHANNEL_NAMES]
    else:
        values = [float(value) for value in raw]
    if len(values) != len(CHANNEL_NAMES):
        raise ValueError(f"loss.channel_weights must have {len(CHANNEL_NAMES)} values")
    return torch.tensor(values, dtype=torch.float32, device=device).clamp_min(0.0)


def decoder_loss_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    surf: torch.Tensor,
    local_x: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = cfg.get("loss", {}) or {}
    err = (pred - target) ** 2
    valid = mask.float()
    point_weight = valid

    surface_point_weight = float(loss_cfg.get("surface_point_weight", 0.0))
    if surface_point_weight:
        point_weight = point_weight * (1.0 + surface_point_weight * surf.float())

    boundary_layer_weight = float(loss_cfg.get("boundary_layer_weight", 0.0))
    if boundary_layer_weight:
        bl_mask = local_x[..., -1].detach().float().clamp_min(0.0)
        point_weight = point_weight * (1.0 + boundary_layer_weight * bl_mask)

    denom = point_weight.sum().clamp_min(1.0)
    channel_mse = (err * point_weight.unsqueeze(-1)).sum(dim=(0, 1)) / denom
    weights = _channel_weights(cfg, pred.device)
    field_loss = (channel_mse * weights).sum() / weights.sum().clamp_min(1e-12)

    surf_valid = (surf.bool() & (mask > 0)).float()
    surf_denom = surf_valid.sum().clamp_min(1.0)
    surface_p_loss = (err[..., 2] * surf_valid).sum() / surf_denom
    total = field_loss + float(loss_cfg.get("surface_pressure_weight", 0.0)) * surface_p_loss

    components = {
        "loss": float(total.detach().cpu()),
        "field_loss": float(field_loss.detach().cpu()),
        "surface_p_loss": float(surface_p_loss.detach().cpu()),
        "ux_mse": float(channel_mse[0].detach().cpu()),
        "uy_mse": float(channel_mse[1].detach().cpu()),
        "p_mse": float(channel_mse[2].detach().cpu()),
        "nut_mse": float(channel_mse[3].detach().cpu()),
    }
    return total, components


@torch.no_grad()
def evaluate_decoder_loader(
    cfg: dict[str, Any],
    decoder: MarioDecoder,
    loader: DataLoader,
    latent_map: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    decoder.eval()
    channel_sse = torch.zeros(len(CHANNEL_NAMES), dtype=torch.float64, device=device)
    channel_count = torch.zeros((), dtype=torch.float64, device=device)
    surface_p_sse = torch.zeros((), dtype=torch.float64, device=device)
    surface_p_count = torch.zeros((), dtype=torch.float64, device=device)
    loss_sum = 0.0
    batches = 0

    for batch in loader:
        local_x = batch["decoder_x"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        surf = batch["surf"].to(device, non_blocking=True)
        condition = batch_condition(batch, latent_map, device)
        pred = decoder(local_x, condition)
        loss, _ = decoder_loss_components(pred, target, mask, surf, local_x, cfg)
        err = (pred - target) ** 2
        valid = mask > 0
        channel_sse += (err.double() * valid.unsqueeze(-1)).sum(dim=(0, 1))
        channel_count += valid.sum().double()
        surf_valid = surf.bool() & valid
        surface_p_sse += err[..., 2].double().masked_select(surf_valid).sum()
        surface_p_count += surf_valid.sum().double()
        loss_sum += float(loss.detach().cpu())
        batches += 1

    mse = channel_sse / channel_count.clamp_min(1.0)
    surface_p_mse = surface_p_sse / surface_p_count.clamp_min(1.0)
    normalized = {
        "ux": float(mse[0].detach().cpu()),
        "uy": float(mse[1].detach().cpu()),
        "p": float(mse[2].detach().cpu()),
        "nut": float(mse[3].detach().cpu()),
        "surface_p": float(surface_p_mse.detach().cpu()),
    }
    paper_scaled = {
        "ux_x1e_minus_2_table_value": normalized["ux"] * 100.0,
        "uy_x1e_minus_2_table_value": normalized["uy"] * 100.0,
        "p_x1e_minus_2_table_value": normalized["p"] * 100.0,
        "nut_x1e_minus_1_table_value": normalized["nut"] * 10.0,
        "surface_p_x1e_minus_1_table_value": normalized["surface_p"] * 10.0,
    }
    return {
        "loss": loss_sum / max(batches, 1),
        "normalized_mse": normalized,
        "paper_table_scaled_mse": paper_scaled,
        "batches": batches,
        "points": int(channel_count.detach().cpu()),
        "surface_points": int(surface_p_count.detach().cpu()),
    }


def validation_score(metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    val_cfg = cfg.get("validation", {}) or {}
    metric = str(val_cfg.get("metric", "paper_table_ratio"))
    if metric == "loss":
        return float(metrics["loss"])
    if metric == "normalized_mse_mean":
        values = metrics["normalized_mse"]
        return float(sum(float(values[name]) for name in CHANNEL_NAMES) / len(CHANNEL_NAMES))
    if metric != "paper_table_ratio":
        raise ValueError(f"Unknown validation.metric={metric!r}")

    scaled = metrics["paper_table_scaled_mse"]
    targets = dict(PAPER_TABLE_TARGETS)
    targets.update({key: float(value) for key, value in (val_cfg.get("paper_targets", {}) or {}).items()})
    weights = {name: 1.0 for name in [*CHANNEL_NAMES, "surface_p"]}
    weights.update({key: float(value) for key, value in (val_cfg.get("score_weights", {}) or {}).items()})
    key_map = {
        "ux": "ux_x1e_minus_2_table_value",
        "uy": "uy_x1e_minus_2_table_value",
        "p": "p_x1e_minus_2_table_value",
        "nut": "nut_x1e_minus_1_table_value",
        "surface_p": "surface_p_x1e_minus_1_table_value",
    }
    numerator = 0.0
    denominator = 0.0
    for name, scaled_key in key_map.items():
        weight = float(weights.get(name, 1.0))
        if weight <= 0.0:
            continue
        numerator += weight * float(scaled[scaled_key]) / max(float(targets[name]), 1e-12)
        denominator += weight
    return numerator / max(denominator, 1e-12)


def train_decoder(
    cfg: dict[str, Any],
    stats: dict[str, np.ndarray],
    encoder: GeometryEncoder,
    device: torch.device,
) -> MarioDecoder:
    latent_path = checkpoint_path(cfg, "train_latents.npz")
    if latent_path.exists():
        latent_map = load_latents(latent_path)
        print(f"loading_latents={latent_path} count={len(latent_map)}", flush=True)
    else:
        latent_map = encode_split(cfg, stats, encoder, cfg["train_split"], device, out_path=latent_path)

    train_names, val_names = split_train_validation_names(cfg)
    print(f"decoder_cases train={len(train_names)} validation={len(val_names)}", flush=True)
    dataset = make_dataset(cfg, cfg["train_split"], stats, deterministic=False, names=train_names)
    loader = make_loader(cfg, dataset, shuffle=True)
    val_cfg = cfg.get("validation", {}) or {}
    val_loader: DataLoader | None = None
    if val_names:
        val_points = int(val_cfg.get("points_per_case", cfg["training"]["points_per_case"]))
        val_dataset = make_dataset(
            cfg,
            cfg["train_split"],
            stats,
            deterministic=True,
            names=val_names,
            points_per_case=val_points,
        )
        val_dataset.set_epoch(0)
        val_loader = make_loader(
            cfg,
            val_dataset,
            shuffle=False,
            batch_size=int(val_cfg.get("batch_size", cfg["training"]["batch_size"])),
        )

    decoder = build_decoder(cfg).to(device)
    weight_decay = float(cfg["decoder"].get("weight_decay", 0.0))
    optimizer_name = str(cfg["decoder"].get("optimizer", "adamw" if weight_decay > 0.0 else "adam")).lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(decoder.parameters(), lr=float(cfg["decoder"]["lr"]), weight_decay=weight_decay)
    path = checkpoint_path(cfg, "decoder_last.pt")
    best_path = checkpoint_path(cfg, "decoder_best.pt")
    history_path = checkpoint_path(cfg, "decoder_validation_history.jsonl")
    start_epoch = 0
    best_score = float("inf")
    best_epoch = 0
    bad_epochs = 0
    if bool(cfg["decoder"].get("resume", True)):
        state = load_checkpoint(path, device)
        if state is not None:
            decoder.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            set_optimizer_lr(optimizer, float(cfg["decoder"]["lr"]))
            for group in optimizer.param_groups:
                group["weight_decay"] = weight_decay
            start_epoch = int(state.get("epoch", 0))
            best_score = float(state.get("best_score", best_score))
            best_epoch = int(state.get("best_epoch", best_epoch))
            print(f"resumed_decoder={path} start_epoch={start_epoch}", flush=True)
            print(f"decoder_lr={optimizer.param_groups[0]['lr']}", flush=True)
        best_state = load_checkpoint(best_path, device)
        if best_state is not None:
            best_score = float(best_state.get("best_score", best_score))
            best_epoch = int(best_state.get("epoch", best_epoch))
            print(f"resumed_decoder_best={best_path} best_epoch={best_epoch} best_score={best_score:.8f}", flush=True)

    epochs = int(cfg["decoder"]["epochs"])
    eval_every = max(int(val_cfg.get("eval_every", 1)), 1)
    patience = int(val_cfg.get("patience", 0))
    min_delta = float(val_cfg.get("min_delta", 0.0))
    grad_clip = cfg["decoder"].get("grad_clip_norm")

    if val_loader is not None and start_epoch > 0 and not best_path.exists():
        initial_metrics = evaluate_decoder_loader(cfg, decoder, val_loader, latent_map, device)
        best_score = validation_score(initial_metrics, cfg)
        best_epoch = start_epoch
        torch.save(
            {
                "model": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": start_epoch,
                "config": cfg,
                "loss": None,
                "train_components": None,
                "validation_metrics": initial_metrics,
                "validation_score": best_score,
                "best_score": best_score,
                "best_epoch": best_epoch,
            },
            best_path,
        )
        append_jsonl(
            history_path,
            {
                "epoch": start_epoch,
                "score": best_score,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "improved": True,
                "train_loss": None,
                "validation": initial_metrics,
                "event": "resume_initial",
            },
        )
        val_mse = initial_metrics["normalized_mse"]
        print(
            "stage=decoder_validation "
            f"epoch={start_epoch} score={best_score:.8f} best={best_score:.8f} improved=True "
            f"ux={val_mse['ux']:.8f} uy={val_mse['uy']:.8f} "
            f"p={val_mse['p']:.8f} nut={val_mse['nut']:.8f} surface_p={val_mse['surface_p']:.8f}",
            flush=True,
        )

    for epoch in range(start_epoch + 1, epochs + 1):
        dataset.set_epoch(epoch)
        decoder.train()
        running = 0.0
        running_components = {
            "field_loss": 0.0,
            "surface_p_loss": 0.0,
            "ux_mse": 0.0,
            "uy_mse": 0.0,
            "p_mse": 0.0,
            "nut_mse": 0.0,
        }
        for batch in loader:
            local_x = batch["decoder_x"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            surf = batch["surf"].to(device, non_blocking=True)
            condition = batch_condition(batch, latent_map, device)
            optimizer.zero_grad(set_to_none=True)
            pred = decoder(local_x, condition)
            loss, components = decoder_loss_components(pred, target, mask, surf, local_x, cfg)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), float(grad_clip))
            optimizer.step()
            running += float(loss.detach().cpu())
            for key in running_components:
                running_components[key] += float(components[key])
        avg = running / max(len(loader), 1)
        train_components = {key: value / max(len(loader), 1) for key, value in running_components.items()}
        print(
            "stage=decoder "
            f"epoch={epoch} train_loss={avg:.8f} "
            f"field_loss={train_components['field_loss']:.8f} "
            f"surface_p_loss={train_components['surface_p_loss']:.8f} "
            f"ux={train_components['ux_mse']:.8f} uy={train_components['uy_mse']:.8f} "
            f"p={train_components['p_mse']:.8f} nut={train_components['nut_mse']:.8f}",
            flush=True,
        )
        validation_metrics = None
        score = None
        improved = False
        if val_loader is not None and (epoch == start_epoch + 1 or epoch % eval_every == 0 or epoch == epochs):
            validation_metrics = evaluate_decoder_loader(cfg, decoder, val_loader, latent_map, device)
            score = validation_score(validation_metrics, cfg)
            improved = score < best_score - min_delta
            if improved:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += eval_every
            append_jsonl(
                history_path,
                {
                    "epoch": epoch,
                    "score": score,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "improved": improved,
                    "train_loss": avg,
                    "validation": validation_metrics,
                },
            )
            val_mse = validation_metrics["normalized_mse"]
            print(
                "stage=decoder_validation "
                f"epoch={epoch} score={score:.8f} best={best_score:.8f} improved={improved} "
                f"ux={val_mse['ux']:.8f} uy={val_mse['uy']:.8f} "
                f"p={val_mse['p']:.8f} nut={val_mse['nut']:.8f} surface_p={val_mse['surface_p']:.8f}",
                flush=True,
            )
        torch.save(
            {
                "model": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": cfg,
                "loss": avg,
                "train_components": train_components,
                "validation_metrics": validation_metrics,
                "validation_score": score,
                "best_score": best_score,
                "best_epoch": best_epoch,
            },
            path,
        )
        if improved:
            torch.save(
                {
                    "model": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": cfg,
                    "loss": avg,
                    "train_components": train_components,
                    "validation_metrics": validation_metrics,
                    "validation_score": score,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
            print(f"stage=decoder_best epoch={epoch} score={best_score:.8f} path={best_path}", flush=True)
        if val_loader is not None and patience > 0 and bad_epochs >= patience:
            print(f"stage=decoder early_stop epoch={epoch} best_epoch={best_epoch} patience={patience}", flush=True)
            break

    if val_loader is not None and bool(val_cfg.get("restore_best", True)) and best_path.exists():
        best_state = load_checkpoint(best_path, device)
        if best_state is not None:
            decoder.load_state_dict(best_state["model"])
            torch.save(best_state, path)
            print(
                f"stage=decoder restored_best epoch={int(best_state.get('epoch', -1))} "
                f"score={float(best_state.get('best_score', float('nan'))):.8f}",
                flush=True,
            )
    return decoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["geometry", "decoder", "all"], default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))
    torch.set_float32_matmul_precision("high")
    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "resolved_config.json", cfg)

    device = device_from_config(cfg["training"])
    print(f"device={device}", flush=True)
    stats = ensure_stats(cfg)
    encoder = build_geometry_encoder(cfg).to(device)
    geom_state = load_checkpoint(checkpoint_path(cfg, "geometry_last.pt"), device)
    if geom_state is not None:
        encoder.load_state_dict(geom_state["model"])

    if args.stage in ("geometry", "all"):
        encoder = train_geometry(cfg, stats, device)
    if args.stage in ("decoder", "all"):
        if not checkpoint_path(cfg, "geometry_last.pt").exists() and args.stage == "decoder":
            raise FileNotFoundError("geometry_last.pt is required for decoder stage")
        decoder = train_decoder(cfg, stats, encoder, device)
        _ = decoder


if __name__ == "__main__":
    main()
