"""Decoder Nd：3 层 MLP，将 latent 解码到输出物理场。"""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Nd: latent [N, latent_dim] -> out [N, C_out]，3 层 MLP。"""

    def __init__(self, latent_dim: int = 32, out_channels: int = 1, hidden: int = 128, layers: int = 3):
        super().__init__()
        widths = [latent_dim] + [hidden] * (layers - 1) + [out_channels]
        seq = []
        for i in range(len(widths) - 1):
            seq.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                seq.append(nn.ReLU())
        self.mlp = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
