from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class FCN(nn.Module):
    """Fully connected network used by each SA-PINN case."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output widths")
        if activation not in {"tanh", "sin", "sine", "relu", "gelu"}:
            raise ValueError(f"unsupported activation: {activation}")
        self.activation = "sin" if activation == "sine" else activation
        stddev = math.sqrt(50.0 / layer_sizes[1])
        self.linears = nn.ModuleList(
            nn.Linear(layer_sizes[index], layer_sizes[index + 1], dtype=dtype)
            for index in range(len(layer_sizes) - 1)
        )
        for linear in self.linears:
            nn.init.trunc_normal_(
                linear.weight,
                mean=0.0,
                std=stddev,
                a=-2.0 * stddev,
                b=2.0 * stddev,
            )
            nn.init.zeros_(linear.bias)

    def _activate(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return torch.nn.functional.gelu(values)
        return getattr(torch, self.activation)(values)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for linear in self.linears[:-1]:
            hidden = self._activate(linear(hidden))
        return self.linears[-1](hidden)


class AttentionWeights(nn.Module):
    """Positive, normalized, per-point self-adaptive loss weights."""

    def __init__(
        self,
        n_points: int,
        dtype: torch.dtype = torch.float64,
        initial_value: float = 0.0,
    ) -> None:
        super().__init__()
        if n_points <= 0:
            raise ValueError("attention requires a positive point count")
        self.alpha = nn.Parameter(
            torch.full((n_points, 1), initial_value, dtype=dtype)
        )

    def forward(self) -> torch.Tensor:
        shifted = self.alpha - self.alpha.detach().max()
        weights = torch.exp(shifted)
        weights = weights / (weights.mean() + torch.finfo(weights.dtype).eps)
        return torch.clamp(weights, min=1.0e-2, max=100.0)


class SAPINN(nn.Module):
    """Self-adaptive PINN with separate attention weights for each loss group."""

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        n_neurons: int,
        activation: str,
        n_pde: int,
        n_data: int,
        n_boundary: int,
        attention_enabled: bool = True,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if min(input_dim, n_layers, n_neurons) <= 0:
            raise ValueError("model dimensions must be positive")
        self.net = FCN(
            [input_dim, *([n_neurons] * n_layers), 1],
            activation=activation,
            dtype=dtype,
        )
        self.att_pde = (
            AttentionWeights(n_pde, dtype=dtype) if attention_enabled and n_pde else None
        )
        self.att_data = (
            AttentionWeights(n_data, dtype=dtype) if attention_enabled and n_data else None
        )
        self.att_boundary = (
            AttentionWeights(n_boundary, dtype=dtype)
            if attention_enabled and n_boundary
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def network_parameters(self):
        return self.net.parameters()

    def attention_parameters(self):
        for attention in (self.att_pde, self.att_data, self.att_boundary):
            if attention is not None:
                yield from attention.parameters()

    def set_attention_trainable(self, trainable: bool) -> None:
        for parameter in self.attention_parameters():
            parameter.requires_grad_(trainable)


def build_model(
    model_config: Mapping,
    point_counts: Mapping[str, int],
    attention_enabled: bool,
    dtype: torch.dtype,
) -> SAPINN:
    return SAPINN(
        input_dim=int(model_config["input_dim"]),
        n_layers=int(model_config["n_layers"]),
        n_neurons=int(model_config["n_neurons"]),
        activation=str(model_config["activation"]),
        n_pde=int(point_counts["pde"]),
        n_data=int(point_counts["data"]),
        n_boundary=int(point_counts["boundary"]),
        attention_enabled=attention_enabled,
        dtype=dtype,
    )


class Equation:
    def residual(
        self, model: SAPINN, coordinates: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class Laplace1D(Equation):
    def __init__(self, diffusion: float = 1.0) -> None:
        self.diffusion = diffusion

    def residual(self, model: SAPINN, coordinates: torch.Tensor) -> torch.Tensor:
        inputs = coordinates.detach().requires_grad_(True)
        prediction = model(inputs)
        first = torch.autograd.grad(
            prediction, inputs, torch.ones_like(prediction), create_graph=True
        )[0]
        second = torch.autograd.grad(
            first, inputs, torch.ones_like(first), create_graph=True
        )[0]
        source = torch.pi**2 * torch.sin(torch.pi * inputs)
        return self.diffusion * second + source


class Helmholtz2D(Equation):
    def __init__(self, wave_number: float = 1.0) -> None:
        self.wave_number = wave_number

    def residual(self, model: SAPINN, coordinates: torch.Tensor) -> torch.Tensor:
        inputs = coordinates.detach().requires_grad_(True)
        prediction = model(inputs)
        gradient = torch.autograd.grad(
            prediction, inputs, torch.ones_like(prediction), create_graph=True
        )[0]
        prediction_xx = torch.autograd.grad(
            gradient[:, 0:1],
            inputs,
            torch.ones_like(gradient[:, 0:1]),
            create_graph=True,
        )[0][:, 0:1]
        prediction_yy = torch.autograd.grad(
            gradient[:, 1:2],
            inputs,
            torch.ones_like(gradient[:, 1:2]),
            create_graph=True,
        )[0][:, 1:2]
        exact = torch.sin(torch.pi * inputs[:, 0:1]) * torch.sin(
            4.0 * torch.pi * inputs[:, 1:2]
        )
        forcing = (
            -torch.pi**2 - (4.0 * torch.pi) ** 2 + self.wave_number**2
        ) * exact
        return (
            prediction_xx
            + prediction_yy
            + self.wave_number**2 * prediction
            - forcing
        )


class Burgers2D(Equation):
    def __init__(self, viscosity: float = 0.01 / math.pi) -> None:
        self.viscosity = viscosity

    def residual(self, model: SAPINN, coordinates: torch.Tensor) -> torch.Tensor:
        inputs = coordinates.detach().requires_grad_(True)
        prediction = model(inputs)
        gradient = torch.autograd.grad(
            prediction, inputs, torch.ones_like(prediction), create_graph=True
        )[0]
        prediction_x = gradient[:, 0:1]
        prediction_t = gradient[:, 1:2]
        prediction_xx = torch.autograd.grad(
            prediction_x,
            inputs,
            torch.ones_like(prediction_x),
            create_graph=True,
        )[0][:, 0:1]
        return (
            prediction_t
            + prediction * prediction_x
            - self.viscosity * prediction_xx
        )


def weighted_mean_square(
    residual: torch.Tensor, attention: AttentionWeights | None
) -> torch.Tensor:
    if attention is None:
        return torch.mean(residual.square())
    weights = attention()
    if weights.shape != residual.shape:
        raise ValueError(
            f"attention shape {tuple(weights.shape)} does not match residual {tuple(residual.shape)}"
        )
    return torch.mean(weights * residual.square())


def loss_components(
    model: SAPINN,
    equation: Equation,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    pde_residual = equation.residual(model, tensors["x_pde"])
    boundary_residual = model(tensors["x_boundary"]) - tensors["u_boundary"]
    if tensors.get("x_data") is not None:
        data_residual = model(tensors["x_data"]) - tensors["u_data"]
        data_loss = weighted_mean_square(data_residual, model.att_data)
    else:
        data_loss = torch.zeros((), dtype=pde_residual.dtype, device=pde_residual.device)
    return {
        "data": data_loss,
        "boundary": weighted_mean_square(boundary_residual, model.att_boundary),
        "pde": weighted_mean_square(pde_residual, model.att_pde),
    }


def weighted_loss(components: Mapping[str, torch.Tensor], weights: Mapping) -> torch.Tensor:
    return (
        float(weights["data"]) * components["data"]
        + float(weights["boundary"]) * components["boundary"]
        + float(weights["pde"]) * components["pde"]
    )
