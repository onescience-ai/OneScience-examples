"""Thin validation wrapper around the official torch-harmonics SFNO class."""

from __future__ import annotations

import torch
from torch import nn
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator

from .config import SFNOConfig


class OfficialSFNOAdapter(nn.Module):
    def __init__(self, config: SFNOConfig) -> None:
        super().__init__()
        config.validate()
        self.expected_shape = (config.channels, config.nlat, config.nlon)
        self.model = SphericalFourierNeuralOperator(
            img_size=(config.nlat, config.nlon),
            grid=config.grid,
            grid_internal=config.grid_internal,
            scale_factor=config.scale_factor,
            in_chans=config.channels,
            out_chans=config.channels,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            use_mlp=True,
            normalization_layer="none",
            residual_prediction=False,
            pos_embed="none",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected [B, C, Nlat, Nlon], got {tuple(inputs.shape)}")
        if tuple(inputs.shape[1:]) != self.expected_shape:
            raise ValueError(
                f"Expected trailing shape {self.expected_shape}, got {tuple(inputs.shape[1:])}"
            )
        if not inputs.is_floating_point():
            raise TypeError("SFNO inputs must be floating point")
        return self.model(inputs.float())
