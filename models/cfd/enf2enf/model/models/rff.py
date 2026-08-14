"""Random Fourier Features（附录 B.1）。

gamma(x) = [cos(Wx), sin(Wx)]，W ~ N(0, sigma_rff^2) 维度 d x d_in。
输出维度 d_gamma = 2 * d。W 默认冻结（train_rff=false）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RandomFourierFeatures(nn.Module):
    def __init__(self, d: int, sigma_rff: float, d_in: int = 2, trainable: bool = False, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(d, d_in, generator=g) * sigma_rff
        self.register_buffer("W", w)
        self.trainable = trainable
        if trainable:
            self.W = nn.Parameter(w)

    @property
    def out_dim(self) -> int:
        return 2 * self.W.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in)
        proj = x @ self.W.t()  # (..., d)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (..., 2d)
