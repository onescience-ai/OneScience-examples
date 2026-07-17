"""Loss functions used by the AirfRANS reproduction."""

from __future__ import annotations

from funcattn_airfrans.utils.metrics import airfrans_lv_ls_loss_torch


def airfrans_loss(pred, target, mask, surf, *, surface_weight: float = 1.0):
    return airfrans_lv_ls_loss_torch(
        pred,
        target,
        mask,
        surf,
        surface_weight=surface_weight,
    )
