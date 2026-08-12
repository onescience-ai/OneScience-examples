"""Independent tiny single-field AtmoRep-style fallback for pipeline validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TinyAtmoRepConfig:
    input_shape: tuple[int, int, int, int] = (4, 1, 8, 8)
    patch_shape: tuple[int, int, int] = (1, 4, 4)
    embed_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    ensemble_size: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


class TinyAtmoRep(nn.Module):
    """Masked-token transformer with four-dimensional token conditioning.

    Inputs use ``[batch, time, variable, latitude, longitude]``. This fallback
    intentionally supports one field only; level is supplied as token metadata.
    """

    def __init__(self, config: TinyAtmoRepConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyAtmoRepConfig()
        time, variables, height, width = self.config.input_shape
        pt, ph, pw = self.config.patch_shape
        if variables != 1:
            raise ValueError("TinyAtmoRep is a single-field fallback (V must equal 1)")
        if time % pt or height % ph or width % pw:
            raise ValueError("input_shape must be divisible by patch_shape")

        self.grid_shape = (time // pt, height // ph, width // pw)
        self.patch_dim = pt * ph * pw
        self.patch_embed = nn.Conv3d(
            1, self.config.embed_dim, kernel_size=self.config.patch_shape,
            stride=self.config.patch_shape,
        )
        self.condition_embed = nn.Sequential(
            nn.Linear(4, self.config.embed_dim), nn.GELU(),
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.embed_dim,
            nhead=self.config.num_heads,
            dim_feedforward=4 * self.config.embed_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, self.config.num_layers)
        self.ensemble_heads = nn.ModuleList(
            nn.Linear(self.config.embed_dim, self.patch_dim)
            for _ in range(self.config.ensemble_size)
        )
        nn.init.normal_(self.mask_token, std=0.02)

    @property
    def num_tokens(self) -> int:
        return math.prod(self.grid_shape)

    def token_conditions(self, batch_size: int, level: float, device: torch.device) -> Tensor:
        """Return normalized [time, level, latitude, longitude] per token."""
        nt, nh, nw = self.grid_shape
        axes = [torch.linspace(-1.0, 1.0, n, device=device) for n in (nt, nh, nw)]
        time, latitude, longitude = torch.meshgrid(*axes, indexing="ij")
        model_level = torch.full_like(time, float(level) / 137.0)
        conditions = torch.stack((time, model_level, latitude, longitude), dim=-1)
        return conditions.reshape(1, self.num_tokens, 4).expand(batch_size, -1, -1)

    def tokenize(self, fields: Tensor) -> Tensor:
        self._validate_fields(fields)
        volume = fields.permute(0, 2, 1, 3, 4)
        pt, ph, pw = self.config.patch_shape
        patches = volume.unfold(2, pt, pt).unfold(3, ph, ph).unfold(4, pw, pw)
        return patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
            fields.shape[0], self.num_tokens, self.patch_dim
        )

    def forward(self, fields: Tensor, mask: Tensor, level: float = 137.0) -> Tensor:
        self._validate_fields(fields)
        if mask.shape != (fields.shape[0], self.num_tokens) or mask.dtype != torch.bool:
            raise ValueError(f"mask must be bool [B, {self.num_tokens}]")
        tokens = self.patch_embed(fields.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)
        conditions = self.token_conditions(fields.shape[0], level, fields.device)
        tokens = tokens + self.condition_embed(conditions)
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        encoded = self.encoder(tokens)
        return torch.stack([head(encoded) for head in self.ensemble_heads], dim=1)

    def _validate_fields(self, fields: Tensor) -> None:
        expected = self.config.input_shape
        if fields.ndim != 5 or tuple(fields.shape[1:]) != expected:
            raise ValueError(f"fields must have shape [B, {expected}], got {tuple(fields.shape)}")


def ensemble_statistical_loss(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
    statistical_weight: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine masked ensemble MSE with ensemble mean/spread statistics."""
    if predictions.ndim != 4 or targets.ndim != 3:
        raise ValueError("predictions must be [B,E,N,P] and targets [B,N,P]")
    selected = mask[:, None, :, None].expand_as(predictions)
    expanded_targets = targets[:, None].expand_as(predictions)
    ensemble_mse = (predictions[selected] - expanded_targets[selected]).square().mean()

    ensemble_mean = predictions.mean(dim=1)
    ensemble_std = predictions.std(dim=1, unbiased=False)
    target_std = targets.std(dim=-1, unbiased=False, keepdim=True).expand_as(targets)
    masked = mask.unsqueeze(-1).expand_as(targets)
    mean_loss = (ensemble_mean[masked] - targets[masked]).square().mean()
    spread_loss = (ensemble_std[masked] - target_std[masked]).square().mean()
    stats_loss = mean_loss + spread_loss
    total = ensemble_mse + statistical_weight * stats_loss
    return total, {
        "ensemble_mse": ensemble_mse.detach(),
        "statistical": stats_loss.detach(),
    }
