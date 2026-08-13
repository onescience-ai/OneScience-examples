"""Independent PyTorch implementation of the DeepONet in arXiv:1910.03193.

The paper is the architectural authority.  No implementation from the official
repository is imported or copied.  ReLU and Xavier-normal initialization are
configurable details used only because the paper leaves them unspecified.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
from torch import Tensor, nn


def _activation(name: str) -> nn.Module:
    choices = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    try:
        return choices[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation {name!r}; choose {sorted(choices)}") from exc


class DenseNetwork(nn.Module):
    """A dense network where ``depth`` counts all Linear layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        width: int,
        activation: str,
        *,
        activate_output: bool,
        dense_bias: bool = True,
        output_bias: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        if min(input_dim, output_dim, width) < 1:
            raise ValueError("input_dim, output_dim and width must be positive")

        layers = []
        current_dim = input_dim
        for layer_index in range(depth):
            is_output = layer_index == depth - 1
            next_dim = output_dim if is_output else width
            layers.append(
                nn.Linear(
                    current_dim,
                    next_dim,
                    bias=output_bias if is_output else dense_bias,
                )
            )
            if not is_output or activate_output:
                layers.append(_activation(activation))
            current_dim = next_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class DeepONet(nn.Module):
    """Stacked or unstacked DeepONet with the paper's branch/trunk fusion."""

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        latent_dim: int,
        *,
        branch_depth: int = 2,
        trunk_depth: int = 3,
        width: int = 40,
        activation: str = "relu",
        stacked: bool = False,
        dense_bias: bool = True,
        branch_output_bias: bool = True,
        global_bias: bool = True,
        initializer: str = "xavier_normal",
    ) -> None:
        super().__init__()
        self.branch_input_dim = int(branch_input_dim)
        self.trunk_input_dim = int(trunk_input_dim)
        self.latent_dim = int(latent_dim)
        self.stacked = bool(stacked)

        branch_kwargs = dict(
            input_dim=self.branch_input_dim,
            output_dim=1 if self.stacked else self.latent_dim,
            depth=branch_depth,
            width=width,
            activation=activation,
            activate_output=False,
            dense_bias=dense_bias,
            output_bias=branch_output_bias,
        )
        if self.stacked:
            self.branch = nn.ModuleList(
                DenseNetwork(**branch_kwargs) for _ in range(self.latent_dim)
            )
        else:
            self.branch = DenseNetwork(**branch_kwargs)

        self.trunk = DenseNetwork(
            input_dim=self.trunk_input_dim,
            output_dim=self.latent_dim,
            depth=trunk_depth,
            width=width,
            activation=activation,
            activate_output=True,
            dense_bias=dense_bias,
            output_bias=dense_bias,
        )
        if global_bias:
            self.output_bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("output_bias", None)
        self.reset_parameters(initializer)

    def reset_parameters(self, initializer: str = "xavier_normal") -> None:
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if initializer == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif initializer == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            else:
                raise ValueError(f"Unsupported initializer {initializer!r}")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_branch(self, branch_inputs: Tensor) -> Tensor:
        if branch_inputs.ndim != 2 or branch_inputs.shape[1] != self.branch_input_dim:
            raise ValueError(
                f"branch input must have shape [N,{self.branch_input_dim}], "
                f"got {tuple(branch_inputs.shape)}"
            )
        if self.stacked:
            return torch.cat([head(branch_inputs) for head in self.branch], dim=-1)
        return self.branch(branch_inputs)

    def forward(self, branch_inputs: Tensor, trunk_inputs: Tensor) -> Tensor:
        if trunk_inputs.ndim != 2 or trunk_inputs.shape[1] != self.trunk_input_dim:
            raise ValueError(
                f"trunk input must have shape [N,{self.trunk_input_dim}], "
                f"got {tuple(trunk_inputs.shape)}"
            )
        if branch_inputs.shape[0] != trunk_inputs.shape[0]:
            raise ValueError("branch and trunk batches must contain the same number of rows")
        branch_features = self.encode_branch(branch_inputs)
        trunk_features = self.trunk(trunk_inputs)
        prediction = torch.sum(branch_features * trunk_features, dim=-1, keepdim=True)
        if self.output_bias is not None:
            prediction = prediction + self.output_bias
        return prediction


class FNNBaseline(nn.Module):
    """Paper baseline that concatenates sensor values and the query coordinate."""

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        *,
        depth: int = 3,
        width: int = 40,
        activation: str = "relu",
        output_bias: bool = True,
        initializer: str = "xavier_normal",
    ) -> None:
        super().__init__()
        self.branch_input_dim = int(branch_input_dim)
        self.trunk_input_dim = int(trunk_input_dim)
        self.network = DenseNetwork(
            input_dim=self.branch_input_dim + self.trunk_input_dim,
            output_dim=1,
            depth=depth,
            width=width,
            activation=activation,
            activate_output=False,
            output_bias=output_bias,
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if initializer == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif initializer == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    raise ValueError(f"Unsupported initializer {initializer!r}")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, branch_inputs: Tensor, trunk_inputs: Tensor) -> Tensor:
        if branch_inputs.ndim != 2 or branch_inputs.shape[1] != self.branch_input_dim:
            raise ValueError("invalid branch input shape")
        if trunk_inputs.ndim != 2 or trunk_inputs.shape[1] != self.trunk_input_dim:
            raise ValueError("invalid trunk input shape")
        return self.network(torch.cat((branch_inputs, trunk_inputs), dim=-1))


def _merged_model_config(config: Mapping[str, Any], experiment: str) -> Dict[str, Any]:
    if experiment not in config.get("experiments", {}):
        raise KeyError(f"Unknown experiment {experiment!r}")
    merged = dict(config.get("model_defaults", {}))
    for key in ("branch_depth", "trunk_depth", "width", "latent_dim"):
        if key in config["experiments"][experiment]:
            merged[key] = config["experiments"][experiment][key]
    return merged


def build_model(
    config: Mapping[str, Any],
    experiment: str,
    variant: str | None = None,
) -> nn.Module:
    """Build a model from the YAML-compatible configuration mapping."""

    experiment_config = config["experiments"][experiment]
    variant_name = variant or experiment_config["default_variant"]
    try:
        variant_config = config["variants"][variant_name]
    except KeyError as exc:
        raise KeyError(f"Unknown model variant {variant_name!r}") from exc
    model_config = _merged_model_config(config, experiment)
    common = dict(
        branch_input_dim=int(experiment_config["sensor_points"]),
        trunk_input_dim=int(experiment_config["trunk_dim"]),
        activation=str(model_config["activation"]),
        initializer=str(model_config["initializer"]),
    )
    if variant_config["architecture"] == "fnn":
        return FNNBaseline(
            **common,
            depth=int(variant_config["depth"]),
            width=int(variant_config["width"]),
            output_bias=bool(variant_config.get("output_bias", True)),
        )
    if variant_config["architecture"] != "deeponet":
        raise ValueError(f"Unsupported architecture {variant_config['architecture']!r}")
    return DeepONet(
        **common,
        latent_dim=int(model_config["latent_dim"]),
        branch_depth=int(model_config["branch_depth"]),
        trunk_depth=int(model_config["trunk_depth"]),
        width=int(model_config["width"]),
        stacked=bool(variant_config["stacked"]),
        dense_bias=bool(model_config.get("dense_bias", True)),
        branch_output_bias=bool(variant_config["branch_output_bias"]),
        global_bias=bool(variant_config["global_bias"]),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["DeepONet", "FNNBaseline", "build_model", "count_parameters"]
