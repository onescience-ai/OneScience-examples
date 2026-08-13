"""Convolutional Neural Operator used by the CNO Navier--Stokes experiment.

The filename is fixed by the reproduction request.  This module implements a
CNO, not a Fourier Neural Operator.  The implementation is written from the
architecture and filter description in Sections 2 and C.1.4 of arXiv:2302.01178.
No source code from the authors' repository is included here.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _design_windowed_sinc(
    num_taps: int,
    resample_factor: int,
    cutoff_denominator: float,
    half_width: float,
) -> Tensor:
    """Construct a finite, symmetric low-pass windowed-sinc filter.

    The paper fixes ``N_tap=12``, ``c_h=0.8`` and a cutoff arbitrarily close
    to the target Nyquist frequency, ``s/2.0001``.  At a high-rate grid used
    for factor-r resampling, this corresponds to a normalized cutoff of
    ``1 / (r * 2.0001)`` cycles per sample.  The exact finite window is not
    specified in the paper; a Kaiser window is used as an explicit,
    configurable approximation to ``scipy.signal.firwin``.
    """
    if num_taps < 2:
        raise ValueError(f"num_taps must be at least 2, got {num_taps}")
    if resample_factor < 1:
        raise ValueError("resample_factor must be positive")
    if cutoff_denominator <= 2.0:
        raise ValueError("cutoff_denominator must be greater than 2")
    if half_width <= 0:
        raise ValueError("half_width must be positive")

    dtype = torch.float64
    positions = torch.arange(num_taps, dtype=dtype) - (num_taps - 1) / 2
    cutoff = 1.0 / (resample_factor * cutoff_denominator)
    ideal = 2.0 * cutoff * torch.sinc(2.0 * cutoff * positions)

    # c_h=0.8 maps to a conventional beta=8.6 window.  Keeping the relation
    # explicit makes the paper-unspecified window choice auditable.
    beta = 8.6 * half_width / 0.8
    window = torch.kaiser_window(num_taps, periodic=False, beta=beta, dtype=dtype)
    kernel = ideal * window
    kernel = kernel / kernel.sum()
    return kernel.to(torch.float32)


class FixedSincResample2d(nn.Module):
    """Separable periodic 2-D windowed-sinc up/downsampling."""

    def __init__(
        self,
        factor: int = 2,
        num_taps: int = 12,
        cutoff_denominator: float = 2.0001,
        half_width: float = 0.8,
    ) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError("factor must be positive")
        self.factor = int(factor)
        kernel = _design_windowed_sinc(
            num_taps=num_taps,
            resample_factor=factor,
            cutoff_denominator=cutoff_denominator,
            half_width=half_width,
        )
        self.register_buffer("kernel", kernel, persistent=True)

    def _filter(self, x: Tensor, gain: float = 1.0) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected BCHW input, got shape {tuple(x.shape)}")
        channels = x.shape[1]
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        taps = int(kernel.numel())
        pad_left = (taps - 1) // 2
        pad_right = taps - 1 - pad_left

        weight_x = (kernel * gain).view(1, 1, 1, taps).repeat(channels, 1, 1, 1)
        x = F.pad(x, (pad_left, pad_right, 0, 0), mode="circular")
        x = F.conv2d(x, weight_x, groups=channels)

        weight_y = kernel.view(1, 1, taps, 1).repeat(channels, 1, 1, 1)
        x = F.pad(x, (0, 0, pad_left, pad_right), mode="circular")
        return F.conv2d(x, weight_y, groups=channels)

    def upsample(self, x: Tensor) -> Tensor:
        if self.factor == 1:
            return x
        batch, channels, height, width = x.shape
        up = x.new_zeros(batch, channels, height * self.factor, width * self.factor)
        up[..., :: self.factor, :: self.factor] = x
        return self._filter(up, gain=float(self.factor * self.factor))

    def downsample(self, x: Tensor) -> Tensor:
        if self.factor == 1:
            return x
        if x.shape[-2] % self.factor or x.shape[-1] % self.factor:
            raise ValueError(
                f"spatial shape {tuple(x.shape[-2:])} is not divisible by {self.factor}"
            )
        return self._filter(x)[..., :: self.factor, :: self.factor]


class BandlimitedActivation(nn.Module):
    """Paper Eq. (2.6): upsample, activate, then low-pass/downsample."""

    def __init__(
        self,
        upsampling_factor: int = 2,
        num_taps: int = 12,
        cutoff_denominator: float = 2.0001,
        half_width: float = 0.8,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)
        self.resampler = FixedSincResample2d(
            factor=upsampling_factor,
            num_taps=num_taps,
            cutoff_denominator=cutoff_denominator,
            half_width=half_width,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.resampler.upsample(x)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        return self.resampler.downsample(x)


def _periodic_conv(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv2d:
    if kernel_size % 2 != 1:
        raise ValueError("CNO convolution kernel_size must be odd")
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        padding_mode="circular",
    )


class CNOBlock(nn.Module):
    """Physical-space convolution followed by bandlimited activation/resampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        resample: Literal["same", "up", "down"],
        activation_kwargs: dict,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.resample = resample
        self.conv = _periodic_conv(in_channels, out_channels, kernel_size)
        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.activation = BandlimitedActivation(**activation_kwargs)
        self.resampler = FixedSincResample2d(
            factor=activation_kwargs["upsampling_factor"],
            num_taps=activation_kwargs["num_taps"],
            cutoff_denominator=activation_kwargs["cutoff_denominator"],
            half_width=activation_kwargs["half_width"],
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.norm(self.conv(x)))
        if self.resample == "down":
            return self.resampler.downsample(x)
        if self.resample == "up":
            return self.resampler.upsample(x)
        return x


class ResidualBlock(nn.Module):
    """Paper Eq. (2.7): identity plus K o Sigma o K."""

    def __init__(self, channels: int, kernel_size: int, activation_kwargs: dict) -> None:
        super().__init__()
        self.conv1 = _periodic_conv(channels, channels, kernel_size)
        self.norm1 = nn.BatchNorm2d(channels)
        self.activation = BandlimitedActivation(**activation_kwargs)
        self.conv2 = _periodic_conv(channels, channels, kernel_size)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.norm1(self.conv1(x))
        residual = self.activation(residual)
        residual = self.norm2(self.conv2(residual))
        return x + residual


class InvariantBlock(nn.Module):
    """Paper Eq. (2.8): Sigma o K at an unchanged representation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation_kwargs: dict,
    ) -> None:
        super().__init__()
        self.conv = _periodic_conv(in_channels, out_channels, kernel_size)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = BandlimitedActivation(**activation_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class LiftProjectBlock(nn.Module):
    """Two convolutions with no BatchNorm, as specified for lift/project."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        kernel_size: int,
        activation_kwargs: dict,
    ) -> None:
        super().__init__()
        self.conv1 = _periodic_conv(in_channels, latent_channels, kernel_size)
        self.activation = BandlimitedActivation(**activation_kwargs)
        self.conv2 = _periodic_conv(latent_channels, out_channels, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.activation(self.conv1(x)))


def _residual_stack(
    channels: int,
    count: int,
    kernel_size: int,
    activation_kwargs: dict,
) -> nn.Module:
    if count == 0:
        return nn.Identity()
    return nn.Sequential(
        *[
            ResidualBlock(channels, kernel_size, activation_kwargs)
            for _ in range(count)
        ]
    )


class CNO2d(nn.Module):
    """Operator U-Net CNO for the paper's 2-D Navier--Stokes experiment."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_width: int = 32,
        levels: int = 3,
        bottleneck_residual_blocks: int = 8,
        intermediate_residual_blocks: int = 1,
        kernel_size: int = 3,
        latent_channels: int = 64,
        activation_upsampling_factor: int = 2,
        filter_taps: int = 12,
        filter_half_width: float = 0.8,
        cutoff_denominator: float = 2.0001,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        if base_width % 2:
            raise ValueError("base_width must be even because lift width is d_e/2")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_width = int(base_width)
        self.levels = int(levels)
        self.required_divisor = 2**levels
        lift_width = base_width // 2
        encoder_widths = [base_width * (2**index) for index in range(levels)]
        activation_kwargs = {
            "upsampling_factor": activation_upsampling_factor,
            "num_taps": filter_taps,
            "cutoff_denominator": cutoff_denominator,
            "half_width": filter_half_width,
            "negative_slope": leaky_relu_slope,
        }

        self.lift = LiftProjectBlock(
            in_channels,
            lift_width,
            latent_channels,
            kernel_size,
            activation_kwargs,
        )

        down_blocks: list[nn.Module] = []
        intermediate_blocks: list[nn.Module] = []
        current_width = lift_width
        for level, next_width in enumerate(encoder_widths):
            down_blocks.append(
                CNOBlock(
                    current_width,
                    next_width,
                    kernel_size,
                    "down",
                    activation_kwargs,
                )
            )
            # The bottleneck has its own N_res,b stack; N_res,i belongs to the
            # genuinely intermediate resolutions only.
            count = intermediate_residual_blocks if level < levels - 1 else 0
            intermediate_blocks.append(
                _residual_stack(next_width, count, kernel_size, activation_kwargs)
            )
            current_width = next_width
        self.encoder = nn.ModuleList(down_blocks)
        self.encoder_residuals = nn.ModuleList(intermediate_blocks)

        self.bottleneck = _residual_stack(
            encoder_widths[-1],
            bottleneck_residual_blocks,
            kernel_size,
            activation_kwargs,
        )

        pre_patch: list[nn.Module] = []
        post_patch: list[nn.Module] = []
        up_blocks: list[nn.Module] = []
        decoder_current = encoder_widths[-1]
        output_widths = list(reversed([lift_width] + encoder_widths[:-1]))
        for next_width in output_widths:
            pre_patch.append(
                InvariantBlock(
                    decoder_current,
                    decoder_current,
                    kernel_size,
                    activation_kwargs,
                )
            )
            post_patch.append(
                InvariantBlock(
                    decoder_current * 2,
                    decoder_current,
                    kernel_size,
                    activation_kwargs,
                )
            )
            up_blocks.append(
                CNOBlock(
                    decoder_current,
                    next_width,
                    kernel_size,
                    "up",
                    activation_kwargs,
                )
            )
            decoder_current = next_width
        self.decoder_pre_patch = nn.ModuleList(pre_patch)
        self.decoder_post_patch = nn.ModuleList(post_patch)
        self.decoder = nn.ModuleList(up_blocks)

        self.project = LiftProjectBlock(
            lift_width * 2,
            out_channels,
            latent_channels,
            kernel_size,
            activation_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"CNO2d expects BCHW input, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {x.shape[1]}"
            )
        height, width = x.shape[-2:]
        if height % self.required_divisor or width % self.required_divisor:
            raise ValueError(
                f"spatial shape {(height, width)} must be divisible by "
                f"2**levels={self.required_divisor}"
            )

        lifted = self.lift(x)
        encoded = lifted
        skips: list[Tensor] = []
        for down, residuals in zip(self.encoder, self.encoder_residuals):
            encoded = residuals(down(encoded))
            skips.append(encoded)

        decoded = self.bottleneck(encoded)
        for pre, post, up, skip in zip(
            self.decoder_pre_patch,
            self.decoder_post_patch,
            self.decoder,
            reversed(skips),
        ):
            decoded = pre(decoded)
            if decoded.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(
                    "decoder/skip spatial mismatch before patching: "
                    f"{tuple(decoded.shape)} versus {tuple(skip.shape)}"
                )
            decoded = post(torch.cat((decoded, skip), dim=1))
            decoded = up(decoded)

        if decoded.shape[-2:] != lifted.shape[-2:]:
            raise RuntimeError(
                f"final decoder/lift mismatch: {decoded.shape} versus {lifted.shape}"
            )
        output = self.project(torch.cat((decoded, lifted), dim=1))
        if output.shape[-2:] != (height, width):
            raise RuntimeError(
                f"CNO changed output grid from {(height, width)} to {output.shape[-2:]}"
            )
        return output


def build_model(model_config: dict) -> CNO2d:
    """Build a CNO2d from the ``model`` section of config.yaml."""
    return CNO2d(**model_config)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["CNO2d", "build_model", "count_trainable_parameters"]
