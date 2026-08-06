"""MfScalingLaws: Multi-Fidelity Scaling Laws for Neural Surrogates in CFD.

Studies the effect of different fidelity data mixing ratios on neural surrogate
scaling laws. Uses standard GNN/MLP models with multi-fidelity experiment configs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FidelityAwareMLP(nn.Module):
    """MLP with fidelity conditioning for multi-fidelity training."""

    def __init__(self, in_dim=7, hidden_dim=128, out_dim=4, num_fidelities=3):
        super().__init__()
        self.fidelity_embedding = nn.Embedding(num_fidelities, hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, fidelity_level):
        # x: (B, N, in_dim), fidelity_level: (B,) int tensor
        f_emb = self.fidelity_embedding(fidelity_level).unsqueeze(1).expand(-1, x.shape[1], -1)
        h = torch.cat([x, f_emb], dim=-1)
        h = self.encoder(h)
        return self.decoder(h)


class FidelityScalingGNN(nn.Module):
    """Simple GNN with fidelity-aware message passing."""

    def __init__(self, in_dim=7, hidden_dim=128, out_dim=4):
        super().__init__()
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index=None):
        # x: (N, in_dim) or (B, N, in_dim)
        if x.dim() == 3:
            x = x.squeeze(0)
        h = self.node_encoder(x)

        # Simple message passing (if edge_index provided)
        if edge_index is not None and edge_index.numel() > 0:
            row, col = edge_index
            edge_feat = torch.cat([h[row], h[col]], dim=-1)
            msg = self.edge_mlp(edge_feat)
            # Aggregate messages
            h = h + torch.zeros_like(h).index_add(0, row, msg)

        return self.node_decoder(h)


class MfScalingLaws(nn.Module):
    """Multi-fidelity neural surrogate with fidelity mixing.

    Supports multiple fidelities with mixing ratio experiments.
    """

    def __init__(self, in_dim=7, hidden_dim=128, out_dim=4, num_fidelities=3):
        super().__init__()
        self.mlp = FidelityAwareMLP(in_dim, hidden_dim, out_dim, num_fidelities)
        self.gnn = FidelityScalingGNN(in_dim, hidden_dim, out_dim)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, fidelity_level=None, edge_index=None):
        if fidelity_level is None:
            fidelity_level = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        mlp_out = self.mlp(x, fidelity_level)
        gnn_out = self.gnn(x.squeeze(0), edge_index)
        if gnn_out.dim() == 2:
            gnn_out = gnn_out.unsqueeze(0)
        return self.fusion_weight * mlp_out + (1 - self.fusion_weight) * gnn_out
