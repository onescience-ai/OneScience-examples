from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class SubNet(nn.Module):
    """Independent neural network assigned to one XPINN subdomain."""

    def __init__(
        self,
        layers: Sequence[int],
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if len(layers) < 2:
            raise ValueError("layers must contain at least an input and an output size")
        if layers[0] != 2 or layers[-1] != 1:
            raise ValueError("XPINN subnetworks must have two inputs and one output")
        if activation not in {"tanh", "sin", "cos"}:
            raise ValueError(f"unsupported activation: {activation}")

        self.activation = activation
        self.linears = nn.ModuleList(
            nn.Linear(layers[index], layers[index + 1], dtype=dtype)
            for index in range(len(layers) - 1)
        )
        # Keep one value per layer for compatibility with the original implementation.
        self.a = nn.ParameterList(
            nn.Parameter(torch.tensor(0.05, dtype=dtype))
            for _ in range(len(layers) - 1)
        )
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        hidden = coordinates
        activation = getattr(torch, self.activation)
        for index, linear in enumerate(self.linears[:-1]):
            hidden = activation(20.0 * self.a[index] * linear(hidden))
        return self.linears[-1](hidden)


class XPINNPoisson2D(nn.Module):
    """Three-subdomain XPINN for the two-dimensional Poisson benchmark."""

    def __init__(
        self, config: Mapping, dtype: torch.dtype = torch.float64
    ) -> None:
        super().__init__()
        try:
            subnetworks = config["subnetworks"]
            domain1 = subnetworks["domain1"]
            domain2 = subnetworks["domain2"]
            domain3 = subnetworks["domain3"]
        except (KeyError, TypeError) as error:
            raise ValueError("model config must define three subnetworks") from error

        self.n1 = SubNet(domain1["layers"], domain1["activation"], dtype=dtype)
        self.n2 = SubNet(domain2["layers"], domain2["activation"], dtype=dtype)
        self.n3 = SubNet(domain3["layers"], domain3["activation"], dtype=dtype)

    @staticmethod
    def _gradient(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            output,
            inputs,
            torch.ones_like(output),
            create_graph=True,
        )[0]

    @staticmethod
    def _source(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) + torch.exp(y)

    def _residual(
        self, network: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = network(torch.cat((x, y), dim=1))
        prediction_x = self._gradient(prediction, x)
        prediction_y = self._gradient(prediction, y)
        prediction_xx = self._gradient(prediction_x, x)
        prediction_yy = self._gradient(prediction_y, y)
        residual = prediction_xx + prediction_yy - self._source(x, y)
        return prediction, residual

    def training_outputs(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        boundary_prediction = self.n1(torch.cat((batch["xb"], batch["yb"]), dim=1))

        _, residual1 = self._residual(self.n1, batch["x1"], batch["y1"])
        _, residual2 = self._residual(self.n2, batch["x2"], batch["y2"])
        _, residual3 = self._residual(self.n3, batch["x3"], batch["y3"])

        interface1_domain1, interface1_residual1 = self._residual(
            self.n1, batch["xi1"], batch["yi1"]
        )
        interface1_domain2, interface1_residual2 = self._residual(
            self.n2, batch["xi1"], batch["yi1"]
        )
        interface2_domain1, interface2_residual1 = self._residual(
            self.n1, batch["xi2"], batch["yi2"]
        )
        interface2_domain3, interface2_residual3 = self._residual(
            self.n3, batch["xi2"], batch["yi2"]
        )

        interface1_average = 0.5 * (interface1_domain1 + interface1_domain2)
        interface2_average = 0.5 * (interface2_domain1 + interface2_domain3)
        return {
            "boundary_prediction": boundary_prediction,
            "residual1": residual1,
            "residual2": residual2,
            "residual3": residual3,
            "interface1_residual": interface1_residual1 - interface1_residual2,
            "interface2_residual": interface2_residual1 - interface2_residual3,
            "interface1_average": interface1_average,
            "interface2_average": interface2_average,
            "interface1_domain1": interface1_domain1,
            "interface1_domain2": interface1_domain2,
            "interface2_domain1": interface2_domain1,
            "interface2_domain3": interface2_domain3,
        }

    def predict(
        self, domain1: torch.Tensor, domain2: torch.Tensor, domain3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.n1(domain1), self.n2(domain2), self.n3(domain3)
