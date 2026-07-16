from __future__ import annotations

import math
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
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GroupEquivariantConv2d(nn.Module):
    """C4/D4 group convolution used by the GFNO lifting and channel mixers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        first_layer: bool = False,
        last_layer: bool = False,
        spectral: bool = False,
        hermitian: bool = False,
        reflection: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.reflection = bool(reflection)
        self.rotation_group_size = 4
        self.group_size = 4 * (1 + int(self.reflection))
        self.kernel_size_y = int(kernel_size)
        self.kernel_size_x = kernel_size // 2 + 1 if hermitian else kernel_size
        self.first_layer = bool(first_layer)
        self.last_layer = bool(last_layer)
        self.hermitian = bool(hermitian)
        dtype = torch.cfloat if spectral else torch.float32

        if self.first_layer or self.last_layer:
            self.weight = nn.Parameter(
                torch.empty(
                    self.out_channels,
                    1,
                    self.in_channels,
                    self.kernel_size_y,
                    self.kernel_size_x,
                    dtype=dtype,
                )
            )
        elif self.hermitian:
            self.weight = nn.ParameterDict(
                {
                    "vertical": nn.Parameter(
                        torch.empty(
                            self.out_channels,
                            1,
                            self.in_channels,
                            self.group_size,
                            self.kernel_size_x - 1,
                            1,
                            dtype=dtype,
                        )
                    ),
                    "positive": nn.Parameter(
                        torch.empty(
                            self.out_channels,
                            1,
                            self.in_channels,
                            self.group_size,
                            self.kernel_size_y,
                            self.kernel_size_x - 1,
                            dtype=dtype,
                        )
                    ),
                    "origin": nn.Parameter(
                        torch.empty(
                            self.out_channels,
                            1,
                            self.in_channels,
                            self.group_size,
                            1,
                            1,
                        )
                    ),
                }
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    self.out_channels,
                    1,
                    self.in_channels,
                    self.group_size,
                    self.kernel_size_y,
                    self.kernel_size_x,
                    dtype=dtype,
                )
            )
        self.base_bias = (
            nn.Parameter(torch.empty(1, self.out_channels, 1, 1))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if isinstance(self.weight, nn.ParameterDict):
            for parameter in self.weight.values():
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.base_bias is not None:
            nn.init.kaiming_uniform_(self.base_bias, a=math.sqrt(5))

    def transformed_weight(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.hermitian:
            vertical = self.weight["vertical"]
            weights = torch.cat(
                (
                    vertical,
                    self.weight["origin"].cfloat(),
                    vertical.flip(dims=(-2,)).conj(),
                ),
                dim=-2,
            )
            weights = torch.cat((weights, self.weight["positive"]), dim=-1)
            weights = torch.cat(
                (weights[..., 1:].conj().rot90(k=2, dims=(-2, -1)), weights),
                dim=-1,
            )
        else:
            weights = self.weight

        if self.first_layer or self.last_layer:
            weights = weights.repeat(1, self.group_size, 1, 1, 1)
            for rotation in range(1, self.rotation_group_size):
                weights[:, rotation] = weights[:, rotation].rot90(
                    k=rotation, dims=(-2, -1)
                )
            if self.reflection:
                weights[:, self.rotation_group_size :] = weights[
                    :, : self.rotation_group_size
                ].flip(dims=(-2,))

            if self.first_layer:
                weights = weights.reshape(
                    self.out_channels * self.group_size,
                    self.in_channels,
                    self.kernel_size_y,
                    self.kernel_size_y,
                )
                bias = (
                    self.base_bias.repeat_interleave(self.group_size, dim=1).reshape(-1)
                    if self.base_bias is not None
                    else None
                )
            else:
                weights = weights.transpose(2, 1).reshape(
                    self.out_channels,
                    self.in_channels * self.group_size,
                    self.kernel_size_y,
                    self.kernel_size_y,
                )
                bias = self.base_bias.reshape(-1) if self.base_bias is not None else None
        else:
            weights = weights.repeat(1, self.group_size, 1, 1, 1, 1)
            for rotation in range(1, self.rotation_group_size):
                weights[:, rotation] = weights[:, rotation - 1].rot90(
                    dims=(-2, -1)
                )
                if self.reflection:
                    weights[:, rotation] = torch.cat(
                        (
                            weights[
                                :, rotation, :, self.rotation_group_size - 1
                            ].unsqueeze(2),
                            weights[
                                :, rotation, :, : self.rotation_group_size - 1
                            ],
                            weights[
                                :, rotation, :, self.rotation_group_size + 1 :
                            ],
                            weights[
                                :, rotation, :, self.rotation_group_size
                            ].unsqueeze(2),
                        ),
                        dim=2,
                    )
                else:
                    weights[:, rotation] = torch.cat(
                        (
                            weights[:, rotation, :, -1].unsqueeze(2),
                            weights[:, rotation, :, :-1],
                        ),
                        dim=2,
                    )
            if self.reflection:
                weights[:, self.rotation_group_size :] = torch.cat(
                    (
                        weights[
                            :, : self.rotation_group_size, :, self.rotation_group_size :
                        ],
                        weights[
                            :, : self.rotation_group_size, :, : self.rotation_group_size
                        ],
                    ),
                    dim=3,
                ).flip(dims=(-2,))

            weights = weights.reshape(
                self.out_channels * self.group_size,
                self.in_channels * self.group_size,
                self.kernel_size_y,
                self.kernel_size_y,
            )
            bias = (
                self.base_bias.repeat_interleave(self.group_size, dim=1).reshape(-1)
                if self.base_bias is not None
                else None
            )

        if self.hermitian:
            weights = weights[..., -self.kernel_size_x :]
        return weights, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, bias = self.transformed_weight()
        return F.conv2d(x, weights, bias=bias)


class GroupSpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        reflection: bool,
    ) -> None:
        super().__init__()
        self.modes = (int(modes), int(modes))
        kernel_size = 2 * max(self.modes) - 1
        self.group_conv = GroupEquivariantConv2d(
            in_channels,
            out_channels,
            kernel_size,
            bias=False,
            spectral=True,
            hermitian=True,
            reflection=reflection,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        modes_y, modes_x = self.modes
        zero_y = height // 2
        start_y = zero_y - modes_y + 1
        stop_y = zero_y + modes_y
        if start_y < 0 or stop_y > height or modes_x > width // 2 + 1:
            raise ValueError(
                f"modes={self.modes} exceed the FFT grid {(height, width)}"
            )

        weights, _ = self.group_conv.transformed_weight()
        weights = weights.transpose(0, 1)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        filtered = x_ft[..., start_y:stop_y, :modes_x]
        out_ft = torch.zeros(
            batch_size,
            weights.shape[1],
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[..., start_y:stop_y, :modes_x] = torch.einsum(
            "bixy,ioxy->boxy", filtered, weights
        )
        return torch.fft.irfft2(
            torch.fft.ifftshift(out_ft, dim=-2), s=(height, width)
        )


class GroupMLP2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        reflection: bool,
        last_layer: bool = False,
    ) -> None:
        super().__init__()
        self.first = GroupEquivariantConv2d(
            in_channels, mid_channels, 1, reflection=reflection
        )
        self.second = GroupEquivariantConv2d(
            mid_channels,
            out_channels,
            1,
            reflection=reflection,
            last_layer=last_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(F.gelu(self.first(x)))


class GroupNorm(nn.Module):
    def __init__(self, width: int, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x = x.reshape(batch_size, -1, self.group_size, height, width)
        x = self.norm(x)
        return x.reshape(batch_size, -1, height, width)


class GFNO(nn.Module):
    """C4/D4 group-equivariant Fourier neural operator for regular 2D grids."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spatial_shape: Sequence[int],
        hidden_dim: int = 64,
        modes: int = 12,
        num_layers: int = 4,
        space_dim: int = 2,
        include_pos: bool = True,
        activation: str = "gelu",
        reflection: bool = False,
        pad_to_multiple: int = 16,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.spatial_shape = tuple(int(value) for value in spatial_shape)
        self.hidden_dim = int(hidden_dim)
        self.modes = int(modes)
        self.num_layers = int(num_layers)
        self.space_dim = int(space_dim)
        self.include_pos = bool(include_pos)
        self.reflection = bool(reflection)
        self.pad_to_multiple = int(pad_to_multiple)
        if len(self.spatial_shape) != 2 or self.space_dim != 2:
            raise ValueError("GFNO supports two-dimensional structured grids")
        if min(self.hidden_dim, self.modes, self.num_layers, self.pad_to_multiple) < 1:
            raise ValueError(
                "hidden_dim, modes, num_layers and pad_to_multiple must be positive"
            )

        feature_dim = self.in_dim + (self.space_dim if self.include_pos else 0)
        self.preprocess = MLP(
            feature_dim, self.hidden_dim * 2, self.hidden_dim, activation
        )
        self.group_size = 4 * (1 + int(self.reflection))
        self.padding = tuple(
            (self.pad_to_multiple - size % self.pad_to_multiple)
            % self.pad_to_multiple
            for size in self.spatial_shape
        )
        augmented_shape = tuple(
            size + padding
            for size, padding in zip(self.spatial_shape, self.padding)
        )
        if self.modes > augmented_shape[0] // 2:
            raise ValueError(
                f"modes={self.modes} exceed padded grid {augmented_shape}"
            )

        self.lifting = GroupEquivariantConv2d(
            self.hidden_dim,
            self.hidden_dim,
            1,
            reflection=self.reflection,
            first_layer=True,
        )
        self.spectral_layers = nn.ModuleList(
            GroupSpectralConv2d(
                self.hidden_dim,
                self.hidden_dim,
                self.modes,
                self.reflection,
            )
            for _ in range(self.num_layers)
        )
        self.mlp_layers = nn.ModuleList(
            GroupMLP2d(
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                self.reflection,
            )
            for _ in range(self.num_layers)
        )
        self.residual_layers = nn.ModuleList(
            GroupEquivariantConv2d(
                self.hidden_dim,
                self.hidden_dim,
                1,
                reflection=self.reflection,
            )
            for _ in range(self.num_layers)
        )
        self.norm = GroupNorm(self.hidden_dim, self.group_size)
        self.output = GroupMLP2d(
            self.hidden_dim,
            self.out_dim,
            self.hidden_dim * 4,
            self.reflection,
            last_layer=True,
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        pad_height, pad_width = self.padding
        return F.pad(x, (0, pad_width, 0, pad_height)) if any(self.padding) else x

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.spatial_shape
        return x[..., :height, :width]

    def forward(
        self, pos: torch.Tensor, field: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, point_count, coordinate_dim = pos.shape
        expected_points = self.spatial_shape[0] * self.spatial_shape[1]
        if point_count != expected_points or coordinate_dim != self.space_dim:
            raise ValueError(
                f"Expected pos [B, {expected_points}, {self.space_dim}], "
                f"got {tuple(pos.shape)}"
            )
        if field is None:
            if self.in_dim:
                raise ValueError("field is required when in_dim > 0")
            features = pos if self.include_pos else pos.new_empty(batch_size, point_count, 0)
        else:
            if field.shape != (batch_size, point_count, self.in_dim):
                raise ValueError(
                    f"Expected field [B, {point_count}, {self.in_dim}], "
                    f"got {tuple(field.shape)}"
                )
            features = torch.cat((pos, field), dim=-1) if self.include_pos else field

        x = self.preprocess(features)
        x = x.permute(0, 2, 1).reshape(
            batch_size, self.hidden_dim, *self.spatial_shape
        )
        x = self.lifting(self._pad(x))
        for layer_index, (spectral, mlp, residual) in enumerate(
            zip(self.spectral_layers, self.mlp_layers, self.residual_layers)
        ):
            update = self.norm(spectral(self.norm(x)))
            x = mlp(update) + residual(x)
            if layer_index != self.num_layers - 1:
                x = F.gelu(x)
        x = self.output(self._unpad(x))
        return x.reshape(batch_size, self.out_dim, -1).permute(0, 2, 1)
