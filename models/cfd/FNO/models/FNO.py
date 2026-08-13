"""Paper-driven Fourier Neural Operator for 2-D Navier--Stokes rollout.

This is an independent implementation of Equations (2), (4), and (5) in
arXiv:2010.08895.  It does not copy the authors' repository implementation.
The public interface is channel-last because the ten history frames are the
input function features; Fourier blocks operate channel-first internally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class SpectralConv2d(nn.Module):
    """Truncated 2-D Fourier integral operator on a periodic grid."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        fft_norm: str = "backward",
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, modes1, modes2) <= 0:
            raise ValueError("channels and retained Fourier modes must be positive")
        if fft_norm not in {"backward", "forward", "ortho"}:
            raise ValueError(f"Unsupported FFT normalization: {fft_norm}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.fft_norm = fft_norm

        # MISSING in the paper: exact complex-weight initialization.  The
        # scale is explicit and seed-controlled by the caller's torch seed.
        scale = 1.0 / (self.in_channels * self.out_channels)
        shape = (self.in_channels, self.out_channels, self.modes1, self.modes2)
        self.weight_positive = nn.Parameter(
            scale * torch.complex(torch.rand(shape), torch.rand(shape))
        )
        self.weight_negative = nn.Parameter(
            scale * torch.complex(torch.rand(shape), torch.rand(shape))
        )

    @staticmethod
    def _multiply_modes(inputs: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"SpectralConv2d expects [B,C,H,W], got {inputs.shape}")
        batch, channels, height, width = inputs.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, received {channels}"
            )
        if 2 * self.modes1 > height:
            raise ValueError(
                f"modes1={self.modes1} overlaps positive/negative bands for H={height}"
            )
        if self.modes2 > width // 2 + 1:
            raise ValueError(
                f"modes2={self.modes2} exceeds rFFT width {width // 2 + 1}"
            )

        spectrum = torch.fft.rfft2(inputs, norm=self.fft_norm)
        output_spectrum = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            device=inputs.device,
            dtype=spectrum.dtype,
        )
        positive_weight = self.weight_positive.to(dtype=spectrum.dtype)
        negative_weight = self.weight_negative.to(dtype=spectrum.dtype)
        output_spectrum[:, :, : self.modes1, : self.modes2] = self._multiply_modes(
            spectrum[:, :, : self.modes1, : self.modes2], positive_weight
        )
        output_spectrum[:, :, -self.modes1 :, : self.modes2] = self._multiply_modes(
            spectrum[:, :, -self.modes1 :, : self.modes2], negative_weight
        )
        return torch.fft.irfft2(
            output_spectrum, s=(height, width), norm=self.fft_norm
        )


class FourierBlock2d(nn.Module):
    """One paper Fourier layer with local W, batch norm, and ReLU."""

    def __init__(self, width: int, modes1: int, modes2: int, fft_norm: str) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2, fft_norm)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(width)
        self.activation = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(
            self.batch_norm(self.spectral(inputs) + self.pointwise(inputs))
        )


class FNO2d(nn.Module):
    """Four-layer FNO-2D mapping ten vorticity frames to the next frame."""

    def __init__(
        self,
        input_channels: int = 10,
        output_channels: int = 1,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        num_layers: int = 4,
        projection_width: int = 128,
        use_grid: bool = True,
        grid_include_endpoint: bool = False,
        expected_resolution: tuple[int, int] = (64, 64),
        fft_norm: str = "backward",
    ) -> None:
        super().__init__()
        if num_layers != 4:
            raise ValueError(
                f"The paper reproduction requires four Fourier layers, got {num_layers}"
            )
        if len(expected_resolution) != 2 or min(expected_resolution) <= 0:
            raise ValueError("expected_resolution must contain two positive dimensions")
        if min(input_channels, output_channels, width, projection_width) <= 0:
            raise ValueError("model widths and channel counts must be positive")

        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.width = int(width)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.num_layers = int(num_layers)
        self.projection_width = int(projection_width)
        self.use_grid = bool(use_grid)
        self.grid_include_endpoint = bool(grid_include_endpoint)
        self.expected_resolution = tuple(int(value) for value in expected_resolution)

        lifting_channels = self.input_channels + (2 if self.use_grid else 0)
        self.lifting = nn.Linear(lifting_channels, self.width)
        self.fourier_blocks = nn.ModuleList(
            [
                FourierBlock2d(self.width, self.modes1, self.modes2, fft_norm)
                for _ in range(self.num_layers)
            ]
        )
        self.projection_hidden = nn.Linear(self.width, self.projection_width)
        self.projection_activation = nn.ReLU()
        self.projection_output = nn.Linear(
            self.projection_width, self.output_channels
        )

    def _grid(self, batch: int, height: int, width: int, inputs: Tensor) -> Tensor:
        if self.grid_include_endpoint:
            x = torch.linspace(0.0, 1.0, height, device=inputs.device, dtype=inputs.dtype)
            y = torch.linspace(0.0, 1.0, width, device=inputs.device, dtype=inputs.dtype)
        else:
            x = torch.arange(height, device=inputs.device, dtype=inputs.dtype) / height
            y = torch.arange(width, device=inputs.device, dtype=inputs.dtype) / width
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"FNO2d expects [B,H,W,T_history], got {inputs.shape}")
        batch, height, width, features = inputs.shape
        if features != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} history channels, received {features}"
            )
        if (height, width) != self.expected_resolution:
            raise ValueError(
                f"Expected resolution {self.expected_resolution}, received {(height, width)}"
            )
        if not inputs.is_floating_point():
            raise TypeError(f"FNO2d expects floating-point input, got {inputs.dtype}")

        lifted_inputs = inputs
        if self.use_grid:
            lifted_inputs = torch.cat(
                (inputs, self._grid(batch, height, width, inputs)), dim=-1
            )
        hidden = self.lifting(lifted_inputs).permute(0, 3, 1, 2).contiguous()
        for block in self.fourier_blocks:
            hidden = block(hidden)
        hidden = hidden.permute(0, 2, 3, 1).contiguous()
        hidden = self.projection_activation(self.projection_hidden(hidden))
        return self.projection_output(hidden)


def build_model_from_config(config: Mapping[str, Any]) -> FNO2d:
    """Construct the exact paper-reproduction model from a parsed YAML mapping."""
    model = config["model"]
    data = config["data"]
    resolution = tuple(int(value) for value in data["resolution"])
    return FNO2d(
        input_channels=int(model["input_channels"]),
        output_channels=int(model["output_channels"]),
        width=int(model["width"]),
        modes1=int(model["modes1"]),
        modes2=int(model["modes2"]),
        num_layers=int(model["num_layers"]),
        projection_width=int(model["projection_width"]),
        use_grid=bool(model["use_grid"]),
        grid_include_endpoint=bool(model.get("grid_include_endpoint", False)),
        expected_resolution=resolution,
        fft_norm=str(model.get("fft_norm", "backward")),
    )
