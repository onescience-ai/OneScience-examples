"""Packed-Ensemble (PE) MLP and baseline MLP models for AirfRANS flow-field regression.

Reproduces arXiv:2312.13403 "Packed-Ensemble Surrogate Models for Fluid Flow
Estimation Around Airfoil Geometries".

Semantics of PackedLinear follow torch-uncertainty:
  - num_estimators M: number of parallel sub-networks (estimators).
  - alpha: width multiplier of each sub-network layer.
  - gamma: number of groups; increasing gamma reduces parameters (sparsity).
  - first=True: input not split by alpha (raw in_features).
  - last=True: output has out_features * M channels (one head per estimator).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PackedLinear(nn.Module):
    """Einsum-based packed linear layer for ensemble sub-networks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_estimators: int,
        alpha: float,
        gamma: int = 1,
        bias: bool = True,
        first: bool = False,
        last: bool = False,
    ) -> None:
        super().__init__()
        self.M = num_estimators
        self.alpha = alpha
        self.gamma = gamma
        self.first = first
        self.last = last
        self.in_features = in_features
        self.out_features = out_features

        # inner dimensions (after alpha scaling); divisibility applies to inner dims
        if not first:
            divisor = num_estimators * gamma
            if in_features % divisor != 0:
                in_features = in_features - (in_features % divisor)
            self.in_features = in_features
        if not last and out_features % (num_estimators * gamma) != 0:
            out_features = out_features - (out_features % (num_estimators * gamma))
            self.out_features = out_features

        inner_in = in_features if first else int(in_features * alpha)
        inner_out = int(out_features * alpha) if not last else out_features
        self.inner_in = inner_in
        self.inner_out = inner_out

        self.weight = nn.Parameter(
            torch.empty(self.M, inner_out, inner_in)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.M, inner_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) or (N, D)
        if self.first:
            # input (B, N, in_features) -> broadcast across estimators
            w = self.weight  # (M, inner_out, in_features)
            out = torch.einsum("bnd,mkd->bmnk", x, w)
            if self.bias is not None:
                out = out + self.bias.view(1, self.M, 1, self.inner_out)
            return out  # (B, M, N, inner_out)
        else:
            # input is already split into alpha channels: (B, M, N, in*alpha)
            w = self.weight  # (M, inner_out, inner_in)
            out = torch.einsum("bmnd,mkd->bmnk", x, w)
            if self.bias is not None:
                out = out + self.bias.view(1, self.M, 1, self.inner_out)
            return out  # (B, M, N, inner_out)


class PEBMlp(nn.Module):
    """Packed-Ensemble Multilayer Perceptron for point-wise flow regression.

    layers: hidden channel list (without input/output). E.g. for the paper's
    PE(8,4,1): layers=(64,64,8,64,64,64,8,64,64).
    """

    def __init__(
        self,
        in_features: int = 7,
        out_features: int = 4,
        layers: tuple[int, ...] = (64, 64, 8, 64, 64, 64, 8, 64, 64),
        num_estimators: int = 8,
        alpha: float = 4.0,
        gamma: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.M = num_estimators
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        def packed(in_f, out_f, first=False, last=False):
            return PackedLinear(
                in_f,
                out_f,
                num_estimators=num_estimators,
                alpha=alpha,
                gamma=gamma,
                first=first,
                last=last,
            )

        # first layer: in_features -> layers[0]*alpha, input not split
        modules = []
        first_lin = packed(in_features, layers[0], first=True)
        modules.append(("pe_first", first_lin))
        modules.append(("act0", nn.ReLU()))
        if dropout > 0:
            modules.append(("drop0", nn.Dropout(p=dropout)))

        prev = layers[0]
        for i, h in enumerate(layers[1:], start=1):
            modules.append((f"pe{i}", packed(prev, h)))
            modules.append((f"act{i}", nn.ReLU()))
            if dropout > 0:
                modules.append((f"drop{i}", nn.Dropout(p=dropout)))
            prev = h

        # last layer: out_features, output = out_features * M (one per estimator)
        modules.append(("pe_last", packed(prev, out_features, last=True)))

        self.net = nn.Sequential()
        for name, m in modules:
            self.net.add_module(name, m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 7) -> (..., M, out)
        orig_shape = x.shape
        flat = x.reshape(-1, self.in_features)
        out = flat
        for name, m in self.net.named_modules():
            pass  # use Sequential directly
        out = self.net[0](flat)  # first packed linear -> (B, M, N, inner)
        # route through remaining modules; handle 4D tensors
        xm = out
        names = [n for n, _ in self.net.named_children()]
        for idx in range(1, len(self.net)):
            m = self.net[idx]
            xm = m(xm)
        # xm: (B, M, N, out)
        return xm


class PackedEnsembleMLP(nn.Module):
    """Cleaner Packed-Ensemble MLP.

    Accepts input (N, 7) or (B, N, 7). Returns (N, M, 4) averaged-friendly
    ensemble outputs.
    """

    def __init__(
        self,
        in_features: int = 7,
        out_features: int = 4,
        layers: tuple[int, ...] = (64, 64, 8, 64, 64, 64, 8, 64, 64),
        num_estimators: int = 8,
        alpha: float = 4.0,
        gamma: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.M = num_estimators

        def pl(in_f, out_f, first=False, last=False):
            return PackedLinear(
                in_f, out_f, num_estimators, alpha, gamma, first=first, last=last
            )

        self.first = pl(in_features, layers[0], first=True)
        self.acts = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        prev = layers[0]
        for h in layers[1:]:
            self.linears.append(pl(prev, h))
            self.acts.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(p=dropout) if dropout > 0 else nn.Identity())
            prev = h
        self.last = pl(prev, out_features, last=True)
        self.last_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in). -> (B, M, N, out)
        # ensure 2D input (N, in) -> (1, N, in)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, in)
        flat = x  # (B, N, in)
        h = self.first(flat)  # (B, M, N, layers[0])
        for lin, act, drop in zip(self.linears, self.acts, self.dropouts):
            h = act(h)
            h = drop(h)
            h = lin(h)
        h = self.last_act(h)
        h = self.last(h)  # (B, M, N, out)
        # reshape back: keep (..., M, out)
        # B = number of leading groups before N
        return h  # (B, M, N, out)


class BaselineMLP(nn.Module):
    """Plain (non-ensemble) MLP baseline with identical architecture."""

    def __init__(
        self,
        in_features: int = 7,
        out_features: int = 4,
        layers: tuple[int, ...] = (64, 64, 8, 64, 64, 64, 8, 64, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [in_features] + list(layers) + [out_features]
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                blocks.append(nn.ReLU())
                if dropout > 0:
                    blocks.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.in_features if hasattr(self, "in_features") else x.shape[-1])
        out = self.net(flat)
        return out


def build_model(model_type: str, **cfg) -> nn.Module:
    if model_type == "pe_mlp":
        return PackedEnsembleMLP(**cfg)
    elif model_type == "mlp":
        # BaselineMLP doesn't take ensemble params
        allowed = {k: cfg[k] for k in ("in_features", "out_features", "layers", "dropout") if k in cfg}
        return BaselineMLP(**allowed)
    else:
        raise ValueError(f"Unknown model_type {model_type}")


def model_factory(
    model_type: str,
    in_features: int = 7,
    out_features: int = 4,
    layers=(64, 64, 8, 64, 64, 64, 8, 64, 64),
    num_estimators: int = 8,
    alpha: float = 4.0,
    gamma: int = 1,
    dropout: float = 0.0,
) -> nn.Module:
    return build_model(
        model_type,
        in_features=in_features,
        out_features=out_features,
        layers=layers,
        num_estimators=num_estimators,
        alpha=alpha,
        gamma=gamma,
        dropout=dropout,
    )
