"""GeoANF: Geometric Attention Neural Field for airfoil flow field prediction.

Based on: Xiao L, Zhang M, Chang X. Applied Sciences, 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple ReLU MLP."""
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GeoAttention(nn.Module):
    """Multi-head cross-attention between query points and boundary points."""

    def __init__(self, dim=64, num_heads=8):
        super().__init__()
        self.q_proj = MLP(7, dim)
        self.k_proj = MLP(7, dim)
        self.v_proj = MLP(7, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, query, boundary):
        q = self.q_proj(query)
        k = self.k_proj(boundary)
        v = self.v_proj(boundary)
        g, _ = self.attn(q, k, v)
        return g


class GeoANF(nn.Module):
    """Geometric Attention Neural Field for airfoil flow prediction.

    Input:  boundary_points (B,m,7), query_points (B,n,7)
    Output: flow field (B,n,4) [vx, vy, p, mut]
    """

    def __init__(self, in_dim=7, hidden_dim=64, out_dim=4, num_heads=8):
        super().__init__()
        self.query_encoder = MLP(in_dim, hidden_dim)
        self.geo_attention = GeoAttention(hidden_dim, num_heads)
        self.decoder = MLP(hidden_dim * 2, out_dim, hidden_dim)

    def forward(self, boundary_points, query_points):
        z = self.query_encoder(query_points)          # (B, n, d)
        g = self.geo_attention(query_points, boundary_points)  # (B, n, d)
        z_g = torch.cat([z, g], dim=-1)               # (B, n, 2d)
        return self.decoder(z_g)                       # (B, n, 4)
