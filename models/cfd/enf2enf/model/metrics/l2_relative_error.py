"""评估指标：Mean L2 Relative Error（Section 4 / Table 1 Hyper-elastic）。"""

from __future__ import annotations

import torch


def l2_relative_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """逐样本 L2 relative error 的平均。

    (1/N_test) sum_i ||pred_i - target_i||_2 / ||target_i||_2
    pred/target: (B, N, C) 或 (B, C, N)。
    """
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    num = torch.norm(pred - target, dim=1)
    den = torch.norm(target, dim=1) + 1e-12
    rel = num / den
    return float(rel.mean().item())
