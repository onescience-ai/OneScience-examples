"""Metrics for masked point-field regression and AirfRANS-style ranking."""

from __future__ import annotations

import numpy as np


def relative_l2_numpy(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    if mask is None:
        diff_norm = np.linalg.norm((pred - target).reshape(-1))
        target_norm = np.linalg.norm(target.reshape(-1))
    else:
        mask_arr = np.asarray(mask).reshape(-1, 1)
        diff_norm = np.linalg.norm(((pred - target) * mask_arr).reshape(-1))
        target_norm = np.linalg.norm((target * mask_arr).reshape(-1))
    return float(diff_norm / max(target_norm, 1e-12))


def pressure_force_coefficients(
    pressure: np.ndarray,
    normals: np.ndarray,
    *,
    aoa_degrees: float | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Approximate pressure-only drag/lift coefficients.

    This is not the full paper Eq. 57 because wall shear stress and official
    surface quadrature weights are not available in the local dataset contract.
    """

    p = np.asarray(pressure).reshape(-1)
    n = np.asarray(normals)[:, :2]
    if weights is None:
        w = np.ones_like(p) / max(p.size, 1)
    else:
        w = np.asarray(weights).reshape(-1)
        w = w / max(float(np.sum(np.abs(w))), 1e-12)

    force = -np.sum(p[:, None] * n * w[:, None], axis=0)
    if aoa_degrees is None:
        drag_dir = np.array([1.0, 0.0], dtype=np.float64)
    else:
        theta = np.deg2rad(aoa_degrees)
        drag_dir = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    lift_dir = np.array([-drag_dir[1], drag_dir[0]], dtype=np.float64)
    return float(force @ drag_dir), float(force @ lift_dir)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def spearmanr_numpy(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = _rankdata(np.asarray(a).reshape(-1))
    b_rank = _rankdata(np.asarray(b).reshape(-1))
    a_rank -= a_rank.mean()
    b_rank -= b_rank.mean()
    denom = np.linalg.norm(a_rank) * np.linalg.norm(b_rank)
    return float((a_rank @ b_rank) / max(denom, 1e-12))


def masked_relative_l2_torch(pred, target, mask):
    import torch

    mask = mask.unsqueeze(-1).to(pred.dtype)
    diff = (pred - target) * mask
    tgt = target * mask
    diff_norm = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=-1)
    tgt_norm = torch.linalg.vector_norm(tgt.reshape(tgt.shape[0], -1), dim=-1)
    return (diff_norm / torch.clamp(tgt_norm, min=1e-12)).mean()


def masked_subset_relative_l2_torch(pred, target, mask, selector):
    import torch

    point_mask = (mask.bool() & selector.bool()).unsqueeze(-1).to(pred.dtype)
    has_points = point_mask.reshape(point_mask.shape[0], -1).sum(dim=-1) > 0
    if not bool(torch.any(has_points)):
        return pred.new_tensor(0.0)
    diff = (pred - target) * point_mask
    tgt = target * point_mask
    diff_norm = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=-1)
    tgt_norm = torch.linalg.vector_norm(tgt.reshape(tgt.shape[0], -1), dim=-1)
    rel = diff_norm / torch.clamp(tgt_norm, min=1e-12)
    return rel[has_points].mean()


def airfrans_lv_ls_loss_torch(pred, target, mask, surf, *, surface_weight: float = 1.0):
    volume = masked_subset_relative_l2_torch(pred, target, mask, ~surf.bool())
    surface = masked_subset_relative_l2_torch(pred, target, mask, surf.bool())
    return volume + float(surface_weight) * surface, volume, surface
