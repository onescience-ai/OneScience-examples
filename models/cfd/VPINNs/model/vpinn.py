from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import gamma, jacobi, roots_jacobi


def jacobi_poly(
    degree: int, alpha: float, beta: float, points: np.ndarray | float
) -> np.ndarray:
    return jacobi(degree, alpha, beta)(np.asarray(points, dtype=float))


def gauss_lobatto_jacobi_weights(
    order: int, alpha: float = 0.0, beta: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Lobatto-Jacobi nodes and weights, including both endpoints."""
    if order < 3:
        raise ValueError("Gauss-Lobatto-Jacobi quadrature order must be at least 3")

    interior = roots_jacobi(order - 2, alpha + 1, beta + 1)[0]
    polynomial = jacobi_poly(order - 1, alpha, beta, interior)
    if alpha == 0 and beta == 0:
        weights = 2.0 / ((order - 1) * order * np.square(polynomial))
        left_weight = 2.0 / (
            (order - 1)
            * order
            * np.square(jacobi_poly(order - 1, 0, 0, -1.0))
        )
        right_weight = 2.0 / (
            (order - 1)
            * order
            * np.square(jacobi_poly(order - 1, 0, 0, 1.0))
        )
    else:
        denominator = (
            (order - 1)
            * gamma(order)
            * gamma(alpha + beta + order + 1)
            * np.square(polynomial)
        )
        numerator = (
            2.0 ** (alpha + beta + 1)
            * gamma(alpha + order)
            * gamma(beta + order)
        )
        weights = numerator / denominator
        left_weight = (
            (beta + 1)
            * numerator
            / (
                (order - 1)
                * gamma(order)
                * gamma(alpha + beta + order + 1)
                * np.square(jacobi_poly(order - 1, alpha, beta, -1.0))
            )
        )
        right_weight = (
            (alpha + 1)
            * numerator
            / (
                (order - 1)
                * gamma(order)
                * gamma(alpha + beta + order + 1)
                * np.square(jacobi_poly(order - 1, alpha, beta, 1.0))
            )
        )

    nodes = np.concatenate(([-1.0], interior, [1.0]))
    all_weights = np.concatenate(([left_weight], weights, [right_weight]))
    return nodes, all_weights


def test_function_jacobi(
    degree: int, points: np.ndarray, alpha: float = 0.0, beta: float = 0.0
) -> np.ndarray:
    """Evaluate phi_n = P_(n+1) - P_(n-1), which vanishes at both endpoints."""
    if degree < 1:
        raise ValueError("test function degree must be at least 1")
    return jacobi_poly(degree + 1, alpha, beta, points) - jacobi_poly(
        degree - 1, alpha, beta, points
    )


def d_test_function_jacobi(
    degree: int, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the first and second derivatives of a Jacobi test function."""
    if degree < 1:
        raise ValueError("test function degree must be at least 1")
    points = np.asarray(points, dtype=float)
    if degree == 1:
        first = ((degree + 2) / 2) * jacobi_poly(degree, 1, 1, points)
        second = ((degree + 2) * (degree + 3) / 4) * jacobi_poly(
            degree - 1, 2, 2, points
        )
    elif degree == 2:
        first = ((degree + 2) / 2) * jacobi_poly(
            degree, 1, 1, points
        ) - (degree / 2) * jacobi_poly(degree - 2, 1, 1, points)
        second = ((degree + 2) * (degree + 3) / 4) * jacobi_poly(
            degree - 1, 2, 2, points
        )
    else:
        first = ((degree + 2) / 2) * jacobi_poly(
            degree, 1, 1, points
        ) - (degree / 2) * jacobi_poly(degree - 2, 1, 1, points)
        second = ((degree + 2) * (degree + 3) / 4) * jacobi_poly(
            degree - 1, 2, 2, points
        ) - (degree * (degree + 1) / 4) * jacobi_poly(
            degree - 3, 2, 2, points
        )
    return np.asarray(first), np.asarray(second)


class VPINN(nn.Module):
    """Fully connected VPINN backbone with trainable adaptive activations."""

    def __init__(
        self,
        layers: Sequence[int],
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if len(layers) < 2:
            raise ValueError("layers must contain at least an input and an output size")
        if activation not in {"tanh", "sin", "cos"}:
            raise ValueError(f"unsupported activation: {activation}")

        self.activation = activation
        self.linears = nn.ModuleList(
            nn.Linear(layers[index], layers[index + 1], dtype=dtype)
            for index in range(len(layers) - 1)
        )
        # Keep one value per layer for compatibility with the bundled checkpoint.
        self.a = nn.ParameterList(
            nn.Parameter(torch.tensor(0.05, dtype=dtype))
            for _ in range(len(layers) - 1)
        )
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        activation = getattr(torch, self.activation)
        for index, linear in enumerate(self.linears[:-1]):
            hidden = activation(20.0 * self.a[index] * linear(hidden))
        return self.linears[-1](hidden)


def unpack_checkpoint(checkpoint: Mapping) -> tuple[Mapping[str, torch.Tensor], dict]:
    """Support both legacy raw state dictionaries and metadata checkpoints."""
    if "model_state" in checkpoint:
        return checkpoint["model_state"], dict(checkpoint)
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, {}
    raise ValueError("checkpoint does not contain a valid VPINN state dictionary")


def infer_layers(state: Mapping[str, torch.Tensor]) -> list[int]:
    weight_keys = sorted(
        (key for key in state if key.startswith("linears.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if not weight_keys:
        raise ValueError("checkpoint contains no VPINN linear-layer weights")
    layers = [int(state[weight_keys[0]].shape[1])]
    layers.extend(int(state[key].shape[0]) for key in weight_keys)
    return layers


class VPINNWrapper(nn.Module):
    """Load a VPINN checkpoint and expose it as an inference module."""

    def __init__(
        self,
        weight_path: str | Path,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        resolved_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"invalid VPINN checkpoint: {weight_path}")
        state, metadata = unpack_checkpoint(checkpoint)
        layers = metadata.get("layers") or infer_layers(state)
        self.device = resolved_device
        self.dtype = dtype
        self.net = VPINN(layers, dtype=dtype).to(device=resolved_device, dtype=dtype)
        self.net.load_state_dict(state, strict=True)
        self.net.eval()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs.to(device=self.device, dtype=self.dtype))
