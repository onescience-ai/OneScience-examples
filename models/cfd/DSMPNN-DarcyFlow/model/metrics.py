"""评估指标：RMSE（论文 Eq 3）、MSE、L1。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE = sqrt(1/n * sum((y - yhat)^2)) (论文 Eq 3)。"""
    return torch.sqrt(F.mse_loss(pred, target))


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)
