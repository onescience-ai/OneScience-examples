"""Paper-faithful PointNet architecture for PointCFD field regression.

The model follows Figure 5 of Kashefi, Rempe, and Guibas (2021): an input
transform, a feature transform, symmetric max aggregation, and a point-wise
decoder for the nondimensional velocity and pressure fields.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor, nn


class ConvBNReLU(nn.Sequential):
    """A shared point-wise fully connected layer with BN and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class LinearBNReLU(nn.Sequential):
    """A fully connected layer with BN and ReLU."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(
            nn.Linear(in_features, out_features, bias=True),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )


class TransformNet(nn.Module):
    """PointNet transformation network for input or intermediate features."""

    def __init__(self, k: int) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = int(k)
        self.point_mlp = nn.Sequential(
            ConvBNReLU(self.k, 64),
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 1024),
        )
        self.global_mlp = nn.Sequential(
            LinearBNReLU(1024, 512),
            LinearBNReLU(512, 256),
        )
        self.transform = nn.Linear(256, self.k * self.k, bias=True)

        # The paper adopts PointNet's canonical identity initialization.
        nn.init.zeros_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Predict a transform from channel-first features ``[B, k, N]``."""
        if features.ndim != 3 or features.shape[1] != self.k:
            raise ValueError(
                f"TransformNet({self.k}) expects [B,{self.k},N], "
                f"got {tuple(features.shape)}"
            )
        encoded = self.point_mlp(features)
        global_feature = torch.amax(encoded, dim=2)
        transform_delta = self.transform(self.global_mlp(global_feature))
        identity = torch.eye(
            self.k, dtype=features.dtype, device=features.device
        ).reshape(1, self.k * self.k)
        return (transform_delta + identity).reshape(-1, self.k, self.k)


class PointNetCFD(nn.Module):
    """Regress normalized ``(u, v, p)`` at every input point."""

    def __init__(self, input_dim: int = 2, output_dim: int = 3) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self.input_transform = TransformNet(self.input_dim)
        self.input_mlp = nn.Sequential(
            ConvBNReLU(self.input_dim, 64),
            ConvBNReLU(64, 64),
        )
        self.feature_transform = TransformNet(64)
        self.global_mlp = nn.Sequential(
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 1024),
        )
        self.decoder = nn.Sequential(
            ConvBNReLU(64 + 1024, 512),
            ConvBNReLU(512, 256),
            ConvBNReLU(256, 128),
            ConvBNReLU(128, 128),
            nn.Conv1d(128, self.output_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self, points: Tensor, return_transforms: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Run point-wise regression.

        Args:
            points: Physical coordinates shaped ``[batch, points, input_dim]``.
            return_transforms: Also return input and feature transform matrices.
        """
        if points.ndim != 3 or points.shape[-1] != self.input_dim:
            raise ValueError(
                f"PointNetCFD expects [B,N,{self.input_dim}], got {tuple(points.shape)}"
            )

        channel_first = points.transpose(1, 2).contiguous()
        input_transform = self.input_transform(channel_first)
        transformed_points = torch.bmm(points, input_transform)

        local_feature = self.input_mlp(
            transformed_points.transpose(1, 2).contiguous()
        )
        feature_transform = self.feature_transform(local_feature)
        transformed_local = torch.bmm(
            local_feature.transpose(1, 2), feature_transform
        ).transpose(1, 2).contiguous()

        encoded = self.global_mlp(transformed_local)
        global_feature = torch.amax(encoded, dim=2, keepdim=True)
        global_repeated = global_feature.expand(-1, -1, points.shape[1])
        decoded_input = torch.cat((transformed_local, global_repeated), dim=1)
        prediction = self.decoder(decoded_input).transpose(1, 2).contiguous()

        if return_transforms:
            return prediction, input_transform, feature_transform
        return prediction


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated by gradient descent."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
