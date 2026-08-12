from typing import Mapping

import torch
from torch import nn


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if logits.ndim == 5:
        b, outputs, bins, h, w = logits.shape
        logits = logits.reshape(b * outputs, bins, h, w)
        target = target.reshape(b * outputs, h, w)
        if mask is not None:
            mask = mask.reshape(b * outputs, h, w)
    value = nn.functional.cross_entropy(logits, target.long(), reduction="none")
    if mask is not None:
        value = value * mask.float()
        return value.sum() / mask.float().sum().clamp_min(1)
    return value.mean()


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    value = (prediction - target).square()
    if mask is not None:
        value = value * mask.float()
        return value.sum() / mask.float().sum().clamp_min(1)
    return value.mean()


def multitask_loss(outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], weights=None):
    weights = weights or {"precipitation": 1.0, "ground": 1.0, "hrrr": 0.25}
    losses = {
        "precipitation": masked_ce(outputs["precipitation_logits"], targets["precipitation"], targets.get("precipitation_mask")),
        "ground": masked_ce(outputs["ground_logits"], targets["ground"], targets.get("ground_mask")),
        "hrrr": masked_mse(outputs["hrrr_regression"], targets["hrrr"], targets.get("hrrr_mask")),
    }
    total = sum(weights[k] * losses[k] for k in losses)
    return total, losses
