from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: str) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SpectralConv2d(nn.Module):
    """Two-dimensional Fourier convolution over the lowest frequency modes."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _multiply(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        modes1 = min(self.modes1, max(1, height // 2))
        modes2 = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :modes1, :modes2] = self._multiply(
            x_ft[:, :, :modes1, :modes2],
            self.weights1[:, :, :modes1, :modes2],
        )
        out_ft[:, :, -modes1:, :modes2] = self._multiply(
            x_ft[:, :, -modes1:, :modes2],
            self.weights2[:, :, :modes1, :modes2],
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


def _normalization(normtype: str, channels: int) -> nn.Module:
    normalized = normtype.lower()
    if normalized == "bn":
        return nn.BatchNorm2d(channels)
    if normalized == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported normtype: {normtype}")


class DoubleConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        normtype: str = "in",
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            _normalization(normtype, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _normalization(normtype, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Down2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normtype: str) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(in_channels, out_channels, normtype=normtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Up2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool,
        normtype: str,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = None
            mid_channels = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            mid_channels = None
        self.conv = DoubleConv2d(
            in_channels,
            out_channels,
            mid_channels=mid_channels,
            normtype=normtype,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.bilinear:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        else:
            x = self.up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(torch.cat((skip, x), dim=1))


class UNO(nn.Module):
    """Two-dimensional U-shaped neural operator for regular CFD grids.

    ``pos`` has shape ``[B, H*W, 2]`` and ``fx`` has shape
    ``[B, H*W, in_dim]``. The returned tensor has shape
    ``[B, H*W, out_dim]``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spatial_shape: Sequence[int],
        hidden_dim: int = 64,
        modes: int = 12,
        space_dim: int = 2,
        include_pos: bool = True,
        normtype: str = "in",
        bilinear: bool = True,
        activation: str = "gelu",
        pad_to_multiple: int = 16,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.spatial_shape = tuple(int(value) for value in spatial_shape)
        self.hidden_dim = int(hidden_dim)
        self.modes = int(modes)
        self.space_dim = int(space_dim)
        self.include_pos = bool(include_pos)
        self.pad_to_multiple = int(pad_to_multiple)
        if len(self.spatial_shape) != 2:
            raise ValueError(f"This standalone UNO supports 2D grids, got {self.spatial_shape}")
        if self.space_dim != 2:
            raise ValueError("space_dim must be 2 for the standalone 2D UNO")
        if min(self.hidden_dim, self.modes, self.pad_to_multiple) < 1:
            raise ValueError("hidden_dim, modes and pad_to_multiple must be positive")

        input_dim = self.in_dim + (self.space_dim if self.include_pos else 0)
        self.preprocess = MLP(input_dim, self.hidden_dim * 2, self.hidden_dim, activation)
        factor = 2 if bilinear else 1

        self.inc = DoubleConv2d(self.hidden_dim, self.hidden_dim, normtype=normtype)
        self.down1 = Down2d(self.hidden_dim, self.hidden_dim * 2, normtype)
        self.down2 = Down2d(self.hidden_dim * 2, self.hidden_dim * 4, normtype)
        self.down3 = Down2d(self.hidden_dim * 4, self.hidden_dim * 8, normtype)
        self.down4 = Down2d(self.hidden_dim * 8, self.hidden_dim * 16 // factor, normtype)

        self.up1 = Up2d(self.hidden_dim * 16, self.hidden_dim * 8 // factor, bilinear, normtype)
        self.up2 = Up2d(self.hidden_dim * 8, self.hidden_dim * 4 // factor, bilinear, normtype)
        self.up3 = Up2d(self.hidden_dim * 4, self.hidden_dim * 2 // factor, bilinear, normtype)
        self.up4 = Up2d(self.hidden_dim * 2, self.hidden_dim, bilinear, normtype)
        self.outc = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)

        augmented = self._augmented_shape(self.spatial_shape)
        self.padding = tuple(target - source for target, source in zip(augmented, self.spatial_shape))
        down_channels = [
            self.hidden_dim,
            self.hidden_dim * 2,
            self.hidden_dim * 4,
            self.hidden_dim * 8,
            self.hidden_dim * 16 // factor,
        ]
        up_channels = [
            self.hidden_dim * 16 // factor,
            self.hidden_dim * 8 // factor,
            self.hidden_dim * 4 // factor,
            self.hidden_dim * 2 // factor,
            self.hidden_dim,
        ]
        divisors = [2, 4, 8, 16, 32]
        self.spectral_down = nn.ModuleList(
            self._spectral(channels, augmented, divisor)
            for channels, divisor in zip(down_channels, divisors)
        )
        self.pointwise_down = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=1) for channels in down_channels
        )
        self.spectral_up = nn.ModuleList(
            self._spectral(channels, augmented, divisor)
            for channels, divisor in zip(up_channels, reversed(divisors))
        )
        self.pointwise_up = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=1) for channels in up_channels
        )
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, self.out_dim)

    def _augmented_shape(self, shape: Sequence[int]) -> tuple[int, int]:
        return tuple(
            size + (self.pad_to_multiple - size % self.pad_to_multiple) % self.pad_to_multiple
            for size in shape
        )

    def _spectral(
        self,
        channels: int,
        augmented_shape: Sequence[int],
        divisor: int,
    ) -> SpectralConv2d:
        modes = [
            max(1, min(self.modes, max(1, size // divisor)))
            for size in augmented_shape
        ]
        return SpectralConv2d(channels, channels, modes[0], modes[1])

    @staticmethod
    def _operator(
        x: torch.Tensor,
        spectral: nn.Module,
        pointwise: nn.Module,
    ) -> torch.Tensor:
        return F.gelu(spectral(x) + pointwise(x))

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        pad_height, pad_width = self.padding
        return F.pad(x, (0, pad_width, 0, pad_height)) if any(self.padding) else x

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.spatial_shape
        return x[..., :height, :width]

    def forward(self, pos: torch.Tensor, fx: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, point_count, coordinates = pos.shape
        expected_points = self.spatial_shape[0] * self.spatial_shape[1]
        if point_count != expected_points or coordinates != self.space_dim:
            raise ValueError(
                f"Expected pos [B, {expected_points}, {self.space_dim}], got {tuple(pos.shape)}"
            )
        if fx is None:
            if self.in_dim:
                raise ValueError("fx is required when in_dim > 0")
            features = pos if self.include_pos else pos.new_empty(batch_size, point_count, 0)
        else:
            if fx.shape != (batch_size, point_count, self.in_dim):
                raise ValueError(
                    f"Expected fx [B, {point_count}, {self.in_dim}], got {tuple(fx.shape)}"
                )
            features = torch.cat((pos, fx), dim=-1) if self.include_pos else fx

        x = self.preprocess(features)
        x = x.permute(0, 2, 1).reshape(batch_size, self.hidden_dim, *self.spatial_shape)
        x = self._pad(x)

        x1 = self._operator(self.inc(x), self.spectral_down[0], self.pointwise_down[0])
        x2 = self._operator(self.down1(x1), self.spectral_down[1], self.pointwise_down[1])
        x3 = self._operator(self.down2(x2), self.spectral_down[2], self.pointwise_down[2])
        x4 = self._operator(self.down3(x3), self.spectral_down[3], self.pointwise_down[3])
        x5 = self._operator(self.down4(x4), self.spectral_down[4], self.pointwise_down[4])
        x = self._operator(x5, self.spectral_up[0], self.pointwise_up[0])
        x = self._operator(self.up1(x, x4), self.spectral_up[1], self.pointwise_up[1])
        x = self._operator(self.up2(x, x3), self.spectral_up[2], self.pointwise_up[2])
        x = self._operator(self.up3(x, x2), self.spectral_up[3], self.pointwise_up[3])
        x = self._operator(self.up4(x, x1), self.spectral_up[4], self.pointwise_up[4])
        x = self._unpad(self.outc(x))

        x = x.reshape(batch_size, self.hidden_dim, -1).permute(0, 2, 1)
        return self.fc2(F.gelu(self.fc1(x)))
