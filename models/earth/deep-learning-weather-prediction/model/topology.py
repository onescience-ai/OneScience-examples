"""Independent PyTorch cubed-sphere operators adapted from official DLWP-CS facts."""
import torch
from torch import nn
from torch.nn import functional as F


def _vertical_faces(x, p):
    # Official face order: 0-3 equatorial, 4 south pole, 5 north pole.
    a = [
        torch.cat((x[:, :, 4, -p:, :], x[:, :, 0], x[:, :, 5, :p, :]), 2),
        torch.cat((x[:, :, 4, :, -p:].transpose(-1, -2), x[:, :, 1],
                   x[:, :, 5, :, -p:].flip(-1).transpose(-1, -2)), 2),
        torch.cat((x[:, :, 4, :p, :].flip((2, 3)), x[:, :, 2], x[:, :, 5, -p:, :].flip((2, 3))), 2),
        torch.cat((x[:, :, 4, :, :p].flip(-1).transpose(-1, -2), x[:, :, 3],
                   x[:, :, 5, :, :p].flip(2).transpose(-1, -2)), 2),
        torch.cat((x[:, :, 2, :p, :].flip((2, 3)), x[:, :, 4], x[:, :, 0, :p, :]), 2),
        torch.cat((x[:, :, 0, -p:, :], x[:, :, 5], x[:, :, 2, -p:, :].flip((2, 3))), 2),
    ]
    return torch.stack(a, 2)


class CubeSpherePadding2d(nn.Module):
    """Topology-aware p-wide padding for [B,C,6,H,W] tensors.

    ASSUMPTION: the official 0..5 face orientation is used exactly as in
    DLWP/custom.py; north-pole reversal is handled by the convolution below.
    """
    def __init__(self, padding=1):
        super().__init__()
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != 6 or self.padding[0] != self.padding[1]:
            raise ValueError("expected [B,C,6,H,W] and symmetric cubed-sphere padding")
        p = self.padding[0]
        y = _vertical_faces(x, p)
        out = [
            torch.cat((y[:, :, 3, :, -p:], y[:, :, 0], y[:, :, 1, :, :p]), 3),
            torch.cat((y[:, :, 0, :, -p:], y[:, :, 1], y[:, :, 2, :, :p]), 3),
            torch.cat((y[:, :, 1, :, -p:], y[:, :, 2], y[:, :, 3, :, :p]), 3),
            torch.cat((y[:, :, 2, :, -p:], y[:, :, 3], y[:, :, 0, :, :p]), 3),
        ]
        out.append(torch.cat((out[3][:, :, p:2*p, :].flip(2).transpose(-1, -2), y[:, :, 4],
                              out[1][:, :, p:2*p, :].flip(-1).transpose(-1, -2)), 3))
        out.append(torch.cat((out[3][:, :, -2*p:-p, :].flip(-1).transpose(-1, -2), y[:, :, 5],
                              out[1][:, :, -2*p:-p, :].flip(2).transpose(-1, -2)), 3))
        return torch.stack(out, 2)


class CubeSphereConv2d(nn.Module):
    """Per-face convolution: shared equatorial, polar, optional north weights."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 independent_north_pole=False, flip_north_pole=True):
        super().__init__()
        kwargs = dict(kernel_size=kernel_size, padding=padding)
        self.equatorial = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.polar = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.north = nn.Conv2d(in_channels, out_channels, **kwargs) if independent_north_pole else self.polar
        self.flip_north_pole = flip_north_pole

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != 6:
            raise ValueError("expected [B,C,6,H,W]")
        ys = [self.equatorial(x[:, :, f]) for f in range(4)]
        ys.append(self.polar(x[:, :, 4]))
        north = x[:, :, 5].flip(-2) if self.flip_north_pole else x[:, :, 5]
        north = self.north(north)
        ys.append(north.flip(-2) if self.flip_north_pole else north)
        return torch.stack(ys, 2)
