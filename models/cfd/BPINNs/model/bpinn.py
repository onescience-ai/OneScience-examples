from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class FCN(nn.Module):
    """Fully connected network used by the BPINN solution model."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
        stddev: float | None = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output widths")
        if activation not in {"tanh", "sin", "sine", "relu", "gelu"}:
            raise ValueError(f"unsupported activation: {activation}")
        if stddev is None:
            stddev = math.sqrt(50.0 / layer_sizes[1])
        if stddev <= 0:
            raise ValueError("stddev must be positive")

        self.activation = "sin" if activation == "sine" else activation
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


class PhysNet(nn.Module):
    """Physics-informed network with solution and optional parameter outputs."""

    def __init__(
        self,
        input_dim: int = 1,
        n_layers: int = 4,
        n_neurons: int = 50,
        n_out_sol: int = 1,
        n_out_par: int = 0,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if min(input_dim, n_layers, n_neurons, n_out_sol) <= 0 or n_out_par < 0:
            raise ValueError("network dimensions must be positive")
        self.n_out_sol = n_out_sol
        self.n_out_par = n_out_par
        output_dim = n_out_sol + n_out_par
        self.fcn = FCN(
            [input_dim, *([n_neurons] * n_layers), output_dim],
            activation=activation,
            dtype=dtype,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.fcn(inputs)
        solution = outputs[..., : self.n_out_sol]
        parameters = outputs[..., self.n_out_sol :] if self.n_out_par else None
        return solution, parameters

    def predict_u(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)[0]


class BPINN(nn.Module):
    """Physics-informed neural network used by the BPINNs workflow."""

    def __init__(
        self,
        input_dim: int = 1,
        n_layers: int = 4,
        n_neurons: int = 50,
        n_out_sol: int = 1,
        n_out_par: int = 0,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.net = PhysNet(
            input_dim=input_dim,
            n_layers=n_layers,
            n_neurons=n_neurons,
            n_out_sol=n_out_sol,
            n_out_par=n_out_par,
            activation=activation,
            dtype=dtype,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.net(inputs)

    def predict_u(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net.predict_u(inputs)


def build_model(config: Mapping, dtype: torch.dtype = torch.float64) -> BPINN:
    return BPINN(
        input_dim=int(config["input_dim"]),
        n_layers=int(config["n_layers"]),
        n_neurons=int(config["n_neurons"]),
        n_out_sol=int(config["n_out_sol"]),
        n_out_par=int(config["n_out_par"]),
        activation=str(config["activation"]),
        dtype=dtype,
    )


def laplace1d_loss_components(
    model: BPINN,
    x_solution: torch.Tensor,
    u_solution: torch.Tensor,
    x_boundary: torch.Tensor,
    u_boundary: torch.Tensor,
    x_pde: torch.Tensor,
) -> dict[str, torch.Tensor]:
    solution_prediction = model.predict_u(x_solution)
    boundary_prediction = model.predict_u(x_boundary)

    pde_points = x_pde.detach().requires_grad_(True)
    pde_prediction = model.predict_u(pde_points)
    first_derivative = torch.autograd.grad(
        pde_prediction,
        pde_points,
        torch.ones_like(pde_prediction),
        create_graph=True,
    )[0]
    second_derivative = torch.autograd.grad(
        first_derivative,
        pde_points,
        torch.ones_like(first_derivative),
        create_graph=True,
    )[0]
    source = torch.pi**2 * torch.sin(torch.pi * pde_points)
    residual = second_derivative + source
    return {
        "data": torch.mean((solution_prediction - u_solution).square()),
        "boundary": torch.mean((boundary_prediction - u_boundary).square()),
        "pde": torch.mean(residual.square()),
    }


def weighted_loss(components: Mapping[str, torch.Tensor], weights: Mapping) -> torch.Tensor:
    return (
        float(weights["data"]) * components["data"]
        + float(weights["boundary"]) * components["boundary"]
        + float(weights["pde"]) * components["pde"]
    )


def posterior_predict(
    model: BPINN,
    states: Sequence[Mapping[str, torch.Tensor]],
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute predictive mean and standard deviation from parameter-state samples."""
    if not states:
        raise ValueError("posterior prediction requires at least one parameter state")
    original_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    predictions = []
    try:
        for state in states:
            model.load_state_dict(state, strict=True)
            with torch.no_grad():
                predictions.append(model.predict_u(inputs))
    finally:
        model.load_state_dict(original_state, strict=True)
    samples = torch.stack(predictions)
    return samples.mean(dim=0), samples.std(dim=0, unbiased=False), samples
