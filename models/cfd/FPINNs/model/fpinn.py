from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class FCN(nn.Module):
    """Fully connected trunk used by the fuzzy PINN."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output widths")
        if activation not in {"tanh", "sin", "sine", "relu", "gelu"}:
            raise ValueError(f"unsupported activation: {activation}")
        self.activation = "sin" if activation == "sine" else activation
        self.linears = nn.ModuleList(
            nn.Linear(layer_sizes[index], layer_sizes[index + 1], dtype=dtype)
            for index in range(len(layer_sizes) - 1)
        )

    def _activate(self, values: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return torch.nn.functional.gelu(values)
        return getattr(torch, self.activation)(values)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for linear in self.linears[:-1]:
            hidden = self._activate(linear(hidden))
        return self.linears[-1](hidden)


class FuzzyLayer(nn.Module):
    """Gaussian fuzzy membership rules followed by product inference."""

    def __init__(
        self,
        input_dim: int,
        rule_count: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if min(input_dim, rule_count) <= 0:
            raise ValueError("input_dim and rule_count must be positive")
        self.input_dim = input_dim
        self.rule_count = rule_count
        self.centers = nn.Parameter(torch.empty(rule_count, input_dim, dtype=dtype))
        self.sigma = nn.Parameter(torch.ones(rule_count, input_dim, dtype=dtype))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        differences = inputs[:, None, :] - self.centers[None, :, :]
        variance = self.sigma.square().clamp_min(torch.finfo(inputs.dtype).eps)
        memberships = torch.exp(-differences.square() / variance[None, :, :])
        return memberships.prod(dim=-1)


class FPINNNet(nn.Module):
    """Parallel neural and fuzzy feature branches with a fused linear head."""

    def __init__(
        self,
        hidden_layers: Sequence[int],
        linear_dim: int,
        fuzzy_dim: int,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.fcn = FCN(
            [2, *hidden_layers, linear_dim], activation=activation, dtype=dtype
        )
        self.fuzzy = FuzzyLayer(2, fuzzy_dim, dtype=dtype)
        self.head = nn.Linear(linear_dim + fuzzy_dim, 1, dtype=dtype)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        neural_features = torch.tanh(self.fcn(coordinates))
        fuzzy_features = self.fuzzy(coordinates)
        return self.head(torch.cat((neural_features, fuzzy_features), dim=1))


class FPINNForward(nn.Module):
    """Forward Allen-Cahn solver with fixed PDE parameters."""

    def __init__(
        self, model_config: Mapping, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.net = FPINNNet(
            hidden_layers=model_config["hidden_layers"],
            linear_dim=int(model_config["linear_dim"]),
            fuzzy_dim=int(model_config["fuzzy_dim"]),
            activation=str(model_config["activation"]),
            dtype=dtype,
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.net(coordinates)

    def predict_u(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.forward(coordinates)


class FPINNInverse(FPINNForward):
    """Inverse Allen-Cahn solver with learnable diffusion and reaction values."""

    def __init__(
        self,
        model_config: Mapping,
        initial_lambda_1: float = 1.0,
        initial_lambda_2: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(model_config, dtype=dtype)
        self.lambda_1 = nn.Parameter(torch.tensor(initial_lambda_1, dtype=dtype))
        self.lambda_2 = nn.Parameter(torch.tensor(initial_lambda_2, dtype=dtype))


def build_model(
    task: str,
    model_config: Mapping,
    pde_config: Mapping,
    dtype: torch.dtype,
) -> FPINNForward:
    if task == "forward":
        return FPINNForward(model_config, dtype=dtype)
    if task == "inverse":
        return FPINNInverse(
            model_config,
            initial_lambda_1=float(pde_config["initial_lambda_1"]),
            initial_lambda_2=float(pde_config["initial_lambda_2"]),
            dtype=dtype,
        )
    raise ValueError(f"unsupported task: {task}")


def allen_cahn_residual(
    model: FPINNForward,
    coordinates: torch.Tensor,
    lambda_1: float | torch.Tensor,
    lambda_2: float | torch.Tensor,
) -> torch.Tensor:
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
        - lambda_1 * prediction_xx
        + lambda_2 * (prediction.pow(3) - prediction)
    )


def loss_components(
    model: FPINNForward,
    x_data: torch.Tensor,
    u_data: torch.Tensor,
    x_pde: torch.Tensor,
    lambda_1: float | torch.Tensor,
    lambda_2: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    prediction = model(x_data)
    residual = allen_cahn_residual(model, x_pde, lambda_1, lambda_2)
    return {
        "data": torch.mean((prediction - u_data).square()),
        "pde": torch.mean(residual.square()),
    }


def weighted_loss(components: Mapping[str, torch.Tensor], weights: Mapping) -> torch.Tensor:
    return float(weights["data"]) * components["data"] + float(
        weights["pde"]
    ) * components["pde"]
