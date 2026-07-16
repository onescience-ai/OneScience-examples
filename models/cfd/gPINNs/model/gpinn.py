from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn


class SubNet(nn.Module):
    """Fully connected network with trainable adaptive activations."""

    def __init__(self, layers: Sequence[int], activation: str = "tanh") -> None:
        super().__init__()
        if len(layers) < 2:
            raise ValueError("layers must contain at least an input and an output size")
        if activation not in {"tanh", "sin", "cos"}:
            raise ValueError(f"unsupported activation: {activation}")

        self.activation = activation
        self.linears = nn.ModuleList(
            nn.Linear(layers[index], layers[index + 1], dtype=torch.float64)
            for index in range(len(layers) - 1)
        )
        # Keep one value per layer for compatibility with the published checkpoints.
        self.a = nn.ParameterList(
            nn.Parameter(torch.tensor(0.05, dtype=torch.float64))
            for _ in range(len(layers) - 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for index, linear in enumerate(self.linears[:-1]):
            hidden = 20.0 * self.a[index] * linear(hidden)
            hidden = getattr(torch, self.activation)(hidden)
        return self.linears[-1](hidden)


class gPINN(nn.Module):
    """Gradient-enhanced physics-informed neural network backbone."""

    def __init__(self, layers: Sequence[int], activation: str = "tanh") -> None:
        super().__init__()
        self.net = SubNet(layers, activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class gPINNPoisson2D(nn.Module):
    """2D Poisson model whose output transform enforces zero boundaries."""

    def __init__(self, layers: Sequence[int]) -> None:
        super().__init__()
        self.net = gPINN(layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        x = coordinates[:, 0:1]
        y = coordinates[:, 1:2]
        return x * y * (1.0 - x) * (1.0 - y) * self.net(coordinates)


def exact_poisson1d(x: np.ndarray) -> np.ndarray:
    solution = x + np.sin(8.0 * x) / 8.0
    for frequency in range(1, 5):
        solution += np.sin(frequency * x) / frequency
    return solution


def gpinn_loss_poisson1d(
    model: nn.Module,
    interior: torch.Tensor,
    boundary: torch.Tensor,
    boundary_values: torch.Tensor,
    gradient_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = interior.detach().requires_grad_(True)
    prediction = model(x)
    first = autograd.grad(prediction, x, torch.ones_like(prediction), create_graph=True)[0]
    second = autograd.grad(first, x, torch.ones_like(first), create_graph=True)[0]
    third = autograd.grad(second, x, torch.ones_like(second), create_graph=True)[0]

    source = 8.0 * torch.sin(8.0 * x) + sum(
        frequency * torch.sin(frequency * x) for frequency in range(1, 5)
    )
    source_gradient = (
        torch.cos(x)
        + 4.0 * torch.cos(2.0 * x)
        + 9.0 * torch.cos(3.0 * x)
        + 16.0 * torch.cos(4.0 * x)
        + 64.0 * torch.cos(8.0 * x)
    )

    residual = -second - source
    residual_gradient = -third - source_gradient
    residual_loss = torch.mean(residual.square())
    boundary_loss = torch.mean((model(boundary) - boundary_values).square())
    gradient_loss = torch.mean(residual_gradient.square())
    loss = residual_loss + boundary_loss + gradient_weight * gradient_loss
    return loss, {
        "residual": residual_loss.item(),
        "boundary": boundary_loss.item(),
        "gradient": gradient_loss.item(),
    }


def exact_poisson2d(x: np.ndarray, y: np.ndarray, exponent: float) -> np.ndarray:
    return (16.0 * x * y * (1.0 - x) * (1.0 - y)) ** exponent


def _poisson2d_source(coordinates: torch.Tensor, exponent: float) -> torch.Tensor:
    x = coordinates[:, 0:1]
    y = coordinates[:, 1:2]
    exact = (16.0 * x * y * (1.0 - x) * (1.0 - y)) ** exponent
    exact_gradient = autograd.grad(
        exact, coordinates, torch.ones_like(exact), create_graph=True
    )[0]
    exact_xx = autograd.grad(
        exact_gradient[:, 0:1],
        coordinates,
        torch.ones_like(exact_gradient[:, 0:1]),
        create_graph=True,
    )[0][:, 0:1]
    exact_yy = autograd.grad(
        exact_gradient[:, 1:2],
        coordinates,
        torch.ones_like(exact_gradient[:, 1:2]),
        create_graph=True,
    )[0][:, 1:2]
    return -(exact_xx + exact_yy)


def gpinn_loss_poisson2d(
    model: nn.Module,
    interior: torch.Tensor,
    gradient_weight: float,
    exponent: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    coordinates = interior.detach().requires_grad_(True)
    prediction = model(coordinates)
    prediction_gradient = autograd.grad(
        prediction, coordinates, torch.ones_like(prediction), create_graph=True
    )[0]
    prediction_xx = autograd.grad(
        prediction_gradient[:, 0:1],
        coordinates,
        torch.ones_like(prediction_gradient[:, 0:1]),
        create_graph=True,
    )[0][:, 0:1]
    prediction_yy = autograd.grad(
        prediction_gradient[:, 1:2],
        coordinates,
        torch.ones_like(prediction_gradient[:, 1:2]),
        create_graph=True,
    )[0][:, 1:2]

    residual = prediction_xx + prediction_yy + _poisson2d_source(coordinates, exponent)
    residual_gradient = autograd.grad(
        residual, coordinates, torch.ones_like(residual), create_graph=True
    )[0]
    residual_loss = torch.mean(residual.square())
    gradient_loss = torch.mean(residual_gradient[:, 0:1].square()) + torch.mean(
        residual_gradient[:, 1:2].square()
    )
    loss = residual_loss + gradient_weight * gradient_loss
    return loss, {"residual": residual_loss.item(), "gradient": gradient_loss.item()}


def output_transform_burgers(
    coordinates: torch.Tensor, raw_prediction: torch.Tensor
) -> torch.Tensor:
    x = coordinates[:, 0:1]
    t = coordinates[:, 1:2]
    return (
        (1.0 - x) * (1.0 + x) * (1.0 - torch.exp(-t)) * raw_prediction
        - torch.sin(torch.pi * x)
    )


class gPINNBurgers(nn.Module):
    """Burgers model with hard initial and boundary constraints."""

    def __init__(self, layers: Sequence[int]) -> None:
        super().__init__()
        self.net = gPINN(layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return output_transform_burgers(coordinates, self.net(coordinates))


def burgers_residual(
    model: nn.Module, coordinates: torch.Tensor, create_graph: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = coordinates.detach().requires_grad_(True)
    prediction = model(inputs)
    prediction_gradient = autograd.grad(
        prediction, inputs, torch.ones_like(prediction), create_graph=True
    )[0]
    prediction_x = prediction_gradient[:, 0:1]
    prediction_t = prediction_gradient[:, 1:2]
    prediction_xx = autograd.grad(
        prediction_x,
        inputs,
        torch.ones_like(prediction_x),
        create_graph=create_graph,
    )[0][:, 0:1]
    viscosity = 0.01 / torch.pi
    residual = prediction_t + prediction * prediction_x - viscosity * prediction_xx
    return residual, inputs


def burgers_gpinn_terms(
    model: nn.Module, coordinates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual, inputs = burgers_residual(model, coordinates, create_graph=True)
    residual_gradient = autograd.grad(
        residual, inputs, torch.ones_like(residual), create_graph=True
    )[0]
    return residual, residual_gradient[:, 0:1], residual_gradient[:, 1:2]
