"""Small self-contained blocks approximating the paper's spatial backbone."""

import torch
from torch import nn


class FiLM(nn.Module):
    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.proj = nn.Linear(condition_dim, channels * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(condition).chunk(2, dim=-1)
        return x * (1 + scale[..., None, None]) + shift[..., None, None]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(4, out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(4, out_channels), nn.GELU(),
        )
        self.film = FiLM(out_channels, condition_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.film(self.net(x), condition)


class LongRangeMaxViT(nn.Module):
    """Compact global token mixer; PAPER_GAP: not the unpublished full MaxViT graph."""
    def __init__(self, channels: int, heads: int = 4, blocks: int = 1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(channels, heads, channels * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        return self.encoder(tokens).transpose(1, 2).reshape(b, c, h, w)
