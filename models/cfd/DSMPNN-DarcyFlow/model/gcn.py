"""GCN baseline：6 层 GCNConv，hidden size 378（论文 Sec 3），PyG 实现。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """GCN baseline：6 层隐藏层，size 378。"""

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 378, layers: int = 6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_channels))

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x
