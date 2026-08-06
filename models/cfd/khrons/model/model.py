"""KHRONOS: Kernel-based Neural Surrogate for Multi-fidelity Aerodynamic Prediction.

Uses kernel regression with efficient linear layers for parameter-efficient
flow field prediction on AirfRANS dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelLayer(nn.Module):
    """Kernel regression layer with learnable inducing points."""

    def __init__(self, in_dim=7, hidden_dim=64, num_inducing=200):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(num_inducing, in_dim) * 0.1)
        self.lengthscale = nn.Parameter(torch.ones(1))
        self.weights = nn.Parameter(torch.randn(num_inducing, hidden_dim) * 0.02)

    def forward(self, x):
        # x: (B, N, in_dim)
        dist = torch.cdist(x, self.inducing.unsqueeze(0))  # (B, N, M)
        K = torch.exp(-dist.pow(2) / (2 * self.lengthscale.pow(2) + 1e-8))
        out = K @ self.weights  # (B, N, hidden_dim)
        return out


class KHRONOS(nn.Module):
    """Kernel-based resource-efficient neural surrogate for multi-fidelity CFD."""

    def __init__(self, in_dim=7, hidden_dim=64, out_dim=4):
        super().__init__()
        self.kernel = KernelLayer(in_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        h = self.kernel(x)
        return self.mlp(h)
