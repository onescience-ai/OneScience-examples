import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, channels: int, outputs: int, bins: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, outputs * bins, 1)
        self.outputs, self.bins = outputs, bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return self.proj(x).reshape(b, self.outputs, self.bins, h, w)


class RegressionHead(nn.Module):
    def __init__(self, channels: int, outputs: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, outputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def decode_bins(logits: torch.Tensor, minimum: float = 0.0, maximum: float = 1.0) -> torch.Tensor:
    centers = torch.linspace(minimum, maximum, logits.shape[2], device=logits.device)
    return (logits.softmax(2) * centers[None, None, :, None, None]).sum(2)
