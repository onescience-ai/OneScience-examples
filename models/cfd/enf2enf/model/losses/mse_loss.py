"""损失函数：L^a 与 L^u 全网格 MSE（Section 3.2 Eq.3）。"""

from __future__ import annotations

import torch.nn.functional as F
import torch


def geometry_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L^a: (1/|X_i|) sum_x ||a_i(x) - a_hat_i(x)||^2，逐样本平均。"""
    return F.mse_loss(pred, target)


def field_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L^u: (1/|X_i|) sum_x ||u_i(x) - u_hat_i(x)||^2，逐样本平均。"""
    return F.mse_loss(pred, target)
