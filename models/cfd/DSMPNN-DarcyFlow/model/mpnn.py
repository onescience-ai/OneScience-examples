"""S-MPNN 完整模型（单卡）：Encoder Ne -> h hops Edge-conditioned conv (Kphi) + residual -> Decoder Nd。

论文 Eq (i)-(iv)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dsmpnn.models.encoder import Encoder
from dsmpnn.models.decoder import Decoder
from dsmpnn.models.kernel import Kernel


class S_MPNN(nn.Module):
    """Single-GPU MPNN。"""

    def __init__(self, node_in_channels: int, node_out_channels: int, edge_channels: int,
                 latent_dim: int = 32, hops: int = 8,
                 encoder_hidden: int = 128, decoder_hidden: int = 128,
                 kernel_hidden: int = 128, kernel_layers: int = 2,
                 encoder_layers: int = 3, decoder_layers: int = 3):
        super().__init__()
        self.encoder = Encoder(node_in_channels, latent_dim, encoder_hidden, encoder_layers)
        self.kernel = Kernel(latent_dim, edge_channels, hops, kernel_hidden, kernel_layers)
        self.decoder = Decoder(latent_dim, node_out_channels, decoder_hidden, decoder_layers)
        self.hops = hops

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        latent = self.encoder(x)                                  # Eq (i)
        latent, edge_attr = self.kernel(latent, edge_index, edge_attr)  # Eq (ii)-(iv) x h
        out = self.decoder(latent)                                # Eq (iii)
        return out
