import torch
import torch.nn as nn


class Hypernetwork(nn.Module):
    """Linear hypernetwork h_u: z_u -> phi_u.

    Wrapper kept for clarity; ModulatedINR already contains its own linear
    hypernetwork. This module is used by INFINITY to manage one hypernetwork
    per INR function.
    """

    def __init__(self, latent_dim, phi_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, phi_dim)

    def forward(self, z):
        return self.linear(z)
