"""Edge-conditioned convolution kernel Kϕ。

论文 Eq 1:  v_i^l = 1/|Ei| * sum_{j<=ne} Kϕ(e_ij^(l-1); θ) * v_j^(l-1) + b

Kϕ 是作用于边属性的 MLP，输出一个权重矩阵作用于邻居 latent 特征。
edge_attr -> MLP -> [edge, in_ch, out_ch] 权重矩阵，对邻居特征加权求和并按度归一化。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter, degree


class EdgeConditionedConv(nn.Module):
    """边缘条件图卷积（单次消息传递）。"""

    def __init__(self, in_channels: int, out_channels: int, edge_channels: int,
                 hidden: int = 128, layers: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        widths = [edge_channels] + [hidden] * (layers - 1) + [in_channels * out_channels]
        seq = []
        for i in range(len(widths) - 1):
            seq.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                seq.append(nn.ReLU())
        self.filter_net = nn.Sequential(*seq)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        filters = self.filter_net(edge_attr).view(-1, self.in_channels, self.out_channels)
        neigh = x[dst]
        msg = torch.einsum('ei,eio->eo', neigh, filters)
        deg = degree(src, num_nodes=x.size(0), dtype=torch.float).clamp(min=1.0)
        deg_inv = (1.0 / deg).view(-1, 1)
        out = scatter(msg, src, dim=0, dim_size=x.size(0), reduce='sum') * deg_inv
        return out + self.bias


class Kernel(nn.Module):
    """消息传递网络 Kϕ：执行 h 次 edge-conditioned conv，并在每次 hop 后更新边属性。

    论文 Eq (ii)-(iv) 循环。Eq (iv) 用新节点值差更新边属性；为保持 kernel 输入维度
    闭合（edge_channels 恒定），将 latent 差值经线性投影映射回 edge_channels
    （论文未明确边属性更新维度，此为 ASSUMPTION，见 reproduction_spec）。
    """

    def __init__(self, latent_dim: int, edge_channels: int, hops: int,
                 hidden: int = 128, layers: int = 2):
        super().__init__()
        self.hops = hops
        self.conv = EdgeConditionedConv(latent_dim, latent_dim, edge_channels, hidden, layers)
        self.edge_update_proj = nn.Linear(latent_dim, edge_channels)

    def forward(self, latent: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """执行 h 次消息传递，带残差连接；返回 (更新后 latent, 更新后 edge_attr)。"""
        for _ in range(self.hops):
            res = latent
            latent = self.conv(latent, edge_index, edge_attr) + res
            src = latent[edge_index[0]]
            dst = latent[edge_index[1]]
            edge_attr = self.edge_update_proj(src - dst)
        return latent, edge_attr
