"""Minimal standalone DLWP-CS-style model and training utilities."""
import torch
from torch import nn
from .topology import CubeSphereConv2d, CubeSpherePadding2d


def capped_leaky_relu(x, negative_slope=0.1, cap=10.0):
    """Paper equation (1): 0.1x below zero, x to 10, then capped at 10."""
    return torch.clamp(torch.where(x >= 0, x, negative_slope * x), max=cap)


class _CappedLeakyReLU(nn.Module):
    def forward(self, x):
        return capped_leaky_relu(x)


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            CubeSpherePadding2d(1), CubeSphereConv2d(cin, cout, padding=0),
            nn.GroupNorm(1, cout), _CappedLeakyReLU(), CubeSpherePadding2d(1),
            CubeSphereConv2d(cout, cout, padding=0), nn.GroupNorm(1, cout),
            _CappedLeakyReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DLWPCubeSphereUNet(nn.Module):
    """Small U-Net preserving [B,C,6,H,W], intended for fake-data validation."""
    def __init__(self, in_channels, out_channels, base_channels=8):
        super().__init__()
        self.enc = _Block(in_channels, base_channels)
        self.down = nn.MaxPool2d(2)
        self.mid = _Block(base_channels, base_channels * 2)
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec = _Block(base_channels * 2, base_channels)
        self.out = CubeSphereConv2d(base_channels, out_channels, 1, padding=0)

    def forward(self, x):
        b, c, f, h, w = x.shape
        if f != 6 or h % 2 or w % 2:
            raise ValueError("faces and H/W must be [6] and even")
        skip = self.enc(x)
        pooled = torch.stack([self.down(skip[:, :, i]) for i in range(6)], 2)
        mid = self.mid(pooled)
        up = torch.stack([self.up(mid[:, :, i]) for i in range(6)], 2)
        return self.out(self.dec(torch.cat((up, skip), 1)))


def weighted_mse(pred, target, weights=None):
    err = (pred - target).square()
    return (err * weights).mean() if weights is not None else err.mean()


@torch.no_grad()
def rollout(model, state, steps=2):
    outputs = []
    for _ in range(steps):
        state = model(state)
        outputs.append(state)
    return torch.stack(outputs, 1)
