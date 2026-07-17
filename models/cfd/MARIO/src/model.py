"""MARIO neural field modules."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, num_features: int = 64, sigma: float = 1.0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_features = int(num_features)
        self.sigma = float(sigma)
        basis = torch.randn(self.num_features, self.input_dim) * self.sigma
        self.register_buffer("basis", basis)

    @property
    def output_dim(self) -> int:
        return 2 * self.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * torch.pi * torch.matmul(x, self.basis.t())
        return torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)


class HyperNetwork(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        output_dim: int,
        *,
        depth: int = 3,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if depth <= 1:
            self.net = nn.Linear(condition_dim, output_dim)
            return
        layers: list[nn.Module] = [nn.Linear(condition_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


class ModulatedNeuralField(nn.Module):
    """ReLU MLP with layer-wise additive modulation from a hypernetwork."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
        *,
        hidden_dim: int = 256,
        hidden_layers: int = 5,
        fourier_features: int = 64,
        fourier_sigma: float = 1.0,
        hyper_depth: int = 3,
        hyper_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.fourier = FourierFeatures(input_dim, fourier_features, fourier_sigma)
        self.in_proj = nn.Linear(self.fourier.output_dim, self.hidden_dim)
        self.layers = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(max(self.hidden_layers - 1, 0))
        )
        self.out_proj = nn.Linear(self.hidden_dim, output_dim)
        self.hyper = HyperNetwork(
            condition_dim,
            self.hidden_layers * self.hidden_dim,
            depth=hyper_depth,
            hidden_dim=hyper_hidden_dim,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shifts = self.hyper(condition).view(condition.shape[0], self.hidden_layers, self.hidden_dim)
        h = self.in_proj(self.fourier(x)) + shifts[:, None, 0, :]
        h = F.relu(h)
        for idx, layer in enumerate(self.layers, start=1):
            h = F.relu(layer(h) + shifts[:, None, idx, :])
        return self.out_proj(h)


class GeometryEncoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 8,
        hidden_dim: int = 256,
        hidden_layers: int = 5,
        fourier_features: int = 64,
        fourier_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.field = ModulatedNeuralField(
            input_dim=2,
            output_dim=1,
            condition_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            hyper_depth=1,
            hyper_hidden_dim=hidden_dim,
        )

    def predict_sdf(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.field(coords, latent)

    def encode(
        self,
        coords: torch.Tensor,
        sdf: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        steps: int = 3,
        inner_lr: float = 0.1,
        create_graph: bool = False,
    ) -> torch.Tensor:
        latent = torch.zeros(coords.shape[0], self.latent_dim, device=coords.device, dtype=coords.dtype).requires_grad_(True)
        for _ in range(steps):
            pred = self.predict_sdf(coords, latent)
            loss = masked_mse(pred, sdf, mask)
            grad = torch.autograd.grad(loss, latent, create_graph=create_graph)[0]
            latent = latent - inner_lr * grad
            if not create_graph:
                latent = latent.detach().requires_grad_(True)
        return latent


class MarioDecoder(nn.Module):
    def __init__(
        self,
        *,
        local_dim: int = 6,
        output_dim: int = 4,
        condition_dim: int = 10,
        hidden_dim: int = 256,
        hidden_layers: int = 5,
        fourier_features: int = 64,
        fourier_sigma: float = 1.0,
        hyper_depth: int = 3,
        hyper_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.field = ModulatedNeuralField(
            input_dim=local_dim,
            output_dim=output_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            fourier_features=fourier_features,
            fourier_sigma=fourier_sigma,
            hyper_depth=hyper_depth,
            hyper_hidden_dim=hyper_hidden_dim,
        )

    def forward(self, local_x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.field(local_x, condition)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target) ** 2
    if mask is None:
        return loss.mean()
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0) / loss.shape[-1]


def masked_channel_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (pred - target) ** 2
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    return (loss * mask).sum(dim=(0, 1)) / mask.sum().clamp_min(1.0)
