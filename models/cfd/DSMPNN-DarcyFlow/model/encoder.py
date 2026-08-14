"""Encoder Ne：3 层 MLP，将初始节点属性编码到 latent 空间。"""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Ne: node_attr [N, C_in] -> latent [N, latent_dim]，3 层 MLP。"""

    def __init__(self, in_channels: int, latent_dim: int = 32, hidden: int = 128, layers: int = 3):
        super().__init__()
        widths = [in_channels] + [hidden] * (layers - 1) + [latent_dim]
        seq = []
        for i in range(len(widths) - 1):
            seq.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                seq.append(nn.ReLU())
        self.mlp = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
