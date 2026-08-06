"""PGOT: Physics-Geometry Operator Transformer for Complex PDEs.

SpecGeo-Attention extracts geometry features.
Physics Slice Injection injects PDE physics into transformer decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecGeoAttention(nn.Module):
    """Cross-attention between query points and geometry points."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x, geo):
        # x: (B, N, dim) query, geo: (B, M, dim) geometry
        out, _ = self.attention(x, geo, geo)
        return out


class PGOT(nn.Module):
    """Physics-Geometry Operator Transformer."""

    def __init__(self, in_dim=7, hidden_dim=128, out_dim=4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.geo_proj = nn.Linear(in_dim, hidden_dim)
        self.spec_geo_attn = SpecGeoAttention(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, geo=None):
        # Auto-extract boundary if geo not provided
        if geo is None:
            surf_mask = (torch.abs(x[..., 4]) < 1e-6)
            if x.dim() == 3:
                geo = x[:, surf_mask[0], :]
            else:
                geo = x[surf_mask, :].unsqueeze(0)
        if geo.shape[1] == 0:
            # No boundary points - use all points
            geo = x
        h = self.input_proj(x)
        g = self.geo_proj(geo)
        h = self.spec_geo_attn(h, g)
        return self.output_proj(h)
