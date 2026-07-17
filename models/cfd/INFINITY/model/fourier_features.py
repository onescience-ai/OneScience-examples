import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Fourier feature mapping gamma(x) = [sin(2*pi*B*x); cos(2*pi*B*x)]."""

    def __init__(self, coord_dim, fourier_dim, scale=1.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.fourier_dim = fourier_dim
        self.out_dim = 2 * fourier_dim
        # fixed log-spaced frequencies (Tancik et al. 2020)
        freqs = torch.pow(2.0, torch.arange(fourier_dim, dtype=torch.float32))
        B = freqs.view(1, fourier_dim).repeat(coord_dim, 1) * scale
        self.register_buffer("B", B)  # (coord_dim, fourier_dim)

    def forward(self, x):
        # x: (..., coord_dim)
        xb = torch.matmul(x, self.B)  # (..., fourier_dim)
        return torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
