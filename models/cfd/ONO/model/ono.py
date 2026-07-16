from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _iterative_pinv(matrix: torch.Tensor, iterations: int = 6) -> torch.Tensor:
    absolute = matrix.abs()
    column_norm = absolute.sum(dim=-1).amax()
    row_norm = absolute.sum(dim=-2).amax()
    scale = (column_norm * row_norm).clamp_min(torch.finfo(matrix.dtype).eps)
    inverse = matrix.transpose(-1, -2) / scale
    identity = torch.eye(
        matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    ).reshape(1, 1, matrix.shape[-1], matrix.shape[-1])
    for _ in range(iterations):
        product = matrix @ inverse
        inverse = 0.25 * inverse @ (
            13 * identity
            - product @ (15 * identity - product @ (7 * identity - product))
        )
    return inverse


class NystromAttention(nn.Module):
    """Linear-complexity Nyström approximation of multi-head attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual_conv_kernel: int = 33,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        padding = residual_conv_kernel // 2
        self.residual = nn.Conv2d(
            heads,
            heads,
            kernel_size=(residual_conv_kernel, 1),
            padding=(padding, 0),
            groups=heads,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, point_count, _ = x.shape
        landmarks = self.num_landmarks
        group_size = math.ceil(point_count / landmarks)
        padded_points = landmarks * group_size
        padding = padded_points - point_count
        if padding:
            x = F.pad(x, (0, 0, padding, 0))

        query, key, value = self.to_qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size, padded_points, self.heads, self.dim_head
            ).permute(0, 2, 1, 3)

        query, key, value = map(split_heads, (query, key, value))
        query = query * self.scale
        query_landmarks = query.reshape(
            batch_size, self.heads, landmarks, group_size, self.dim_head
        ).mean(dim=3)
        key_landmarks = key.reshape(
            batch_size, self.heads, landmarks, group_size, self.dim_head
        ).mean(dim=3)

        similarity1 = torch.einsum("bhid,bhjd->bhij", query, key_landmarks)
        similarity2 = torch.einsum(
            "bhid,bhjd->bhij", query_landmarks, key_landmarks
        )
        similarity3 = torch.einsum("bhid,bhjd->bhij", query_landmarks, key)
        attention1 = similarity1.softmax(dim=-1)
        attention2 = similarity2.softmax(dim=-1)
        attention3 = similarity3.softmax(dim=-1)
        attention2_inverse = _iterative_pinv(
            attention2, iterations=self.pinv_iterations
        )
        output = (attention1 @ attention2_inverse) @ (attention3 @ value)
        output = output + self.residual(value)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, padded_points, -1)
        return self.to_out(output[:, -point_count:])


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, point_count, channels = x.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size, point_count, self.heads, self.dim_head
            ).transpose(1, 2)

        query = split_heads(self.query(x)).softmax(dim=-1)
        key = split_heads(self.key(x)).softmax(dim=-1)
        value = split_heads(self.value(x))
        context = key.transpose(-2, -1) @ value
        output = self.dropout((query @ context) / float(point_count) + query)
        output = output.transpose(1, 2).reshape(batch_size, point_count, channels)
        return self.output(output)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.query = nn.Linear(dim, inner_dim, bias=False)
        self.key = nn.Linear(dim, inner_dim, bias=False)
        self.value = nn.Linear(dim, inner_dim, bias=False)
        self.output = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, point_count, _ = x.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size, point_count, self.heads, self.dim_head
            ).transpose(1, 2)

        query = split_heads(self.query(x)).softmax(dim=-1) * self.dim_head**-0.5
        key = split_heads(self.key(x)).softmax(dim=-2)
        value = split_heads(self.value(x))
        context = torch.einsum("bhnd,bhne->bhde", key, value)
        output = torch.einsum("bhnd,bhde->bhne", query, context)
        output = output.transpose(1, 2).reshape(batch_size, point_count, -1)
        return self.output(output)


def _make_attention(
    attn_type: str,
    hidden_dim: int,
    num_heads: int,
    dropout: float,
) -> nn.Module:
    arguments = {
        "dim": hidden_dim,
        "heads": num_heads,
        "dim_head": hidden_dim // num_heads,
        "dropout": dropout,
    }
    if attn_type == "nystrom":
        return NystromAttention(**arguments)
    if attn_type == "linear":
        return LinearAttention(**arguments)
    if attn_type == "selfAttention":
        return SelfAttention(**arguments)
    raise ValueError("attn_type must be nystrom, linear, or selfAttention")


def _safe_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    try:
        factor = torch.linalg.cholesky(matrix)
        if torch.isnan(factor).any():
            raise RuntimeError("Cholesky factor contains NaN")
        return factor
    except RuntimeError as original_error:
        if torch.isnan(matrix).any():
            raise ValueError("Orthogonal feature covariance contains NaN") from original_error
        jitter = 1.0e-6 if matrix.dtype == torch.float32 else 1.0e-8
        stabilized = matrix.clone()
        previous = 0.0
        for exponent in range(10):
            current = jitter * 10**exponent
            stabilized.diagonal().add_(current - previous)
            previous = current
            try:
                factor = torch.linalg.cholesky(stabilized)
                warnings.warn(
                    f"Feature covariance required Cholesky jitter {current}",
                    RuntimeWarning,
                )
                return factor
            except RuntimeError:
                continue
        raise original_error


class OrthogonalNeuralBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        activation: str,
        attn_type: str,
        mlp_ratio: int,
        psi_dim: int,
        out_dim: int,
        last_layer: bool,
        momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.momentum = momentum
        self.register_buffer("feature_cov", torch.zeros(psi_dim, psi_dim))
        self.mu = nn.Parameter(torch.zeros(psi_dim))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = _make_attention(
            attn_type, hidden_dim, num_heads, dropout
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            activation,
        )
        self.projection = nn.Linear(hidden_dim, psi_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.field_output = (
            nn.Linear(hidden_dim, out_dim)
            if last_layer
            else MLP(
                hidden_dim,
                hidden_dim * mlp_ratio,
                hidden_dim,
                activation,
            )
        )

    def forward(
        self, feature: torch.Tensor, field: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.attention(self.norm1(feature)) + feature
        feature = self.mlp(self.norm2(feature)) + feature
        projected = self.projection(feature)
        if self.training:
            covariance = torch.einsum(
                "bnc,bnd->cd", projected, projected
            ) / (projected.shape[0] * projected.shape[1])
            with torch.no_grad():
                self.feature_cov.mul_(self.momentum).add_(
                    covariance, alpha=1.0 - self.momentum
                )
        else:
            covariance = self.feature_cov

        factor = _safe_cholesky(covariance)
        inverse_transpose = torch.linalg.inv(factor).transpose(-2, -1)
        orthogonal = projected @ inverse_transpose
        field = (orthogonal * F.softplus(self.mu)) @ (
            orthogonal.transpose(-2, -1) @ field
        ) + field
        return feature, self.field_output(self.norm3(field))


class ONO(nn.Module):
    """Orthogonal Neural Operator for pointwise physical-field forecasting."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        space_dim: int = 2,
        include_pos: bool = True,
        dropout: float = 0.0,
        activation: str = "gelu",
        mlp_ratio: int = 1,
        attn_type: str = "nystrom",
        psi_dim: int = 8,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.space_dim = int(space_dim)
        self.include_pos = bool(include_pos)
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if min(self.num_layers, self.num_heads, int(mlp_ratio), int(psi_dim)) < 1:
            raise ValueError("num_layers, num_heads, mlp_ratio and psi_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        feature_dim = self.in_dim + (self.space_dim if self.include_pos else 0)
        self.preprocess_feature = MLP(
            feature_dim, self.hidden_dim * 2, self.hidden_dim, activation
        )
        self.preprocess_field = MLP(
            feature_dim, self.hidden_dim * 2, self.hidden_dim, activation
        )
        self.blocks = nn.ModuleList(
            OrthogonalNeuralBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=float(dropout),
                activation=activation,
                attn_type=attn_type,
                mlp_ratio=int(mlp_ratio),
                psi_dim=int(psi_dim),
                out_dim=self.out_dim,
                last_layer=layer == self.num_layers - 1,
            )
            for layer in range(self.num_layers)
        )
        self.placeholder = nn.Parameter(
            torch.rand(self.hidden_dim) / self.hidden_dim
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(
        self, pos: torch.Tensor, field: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, point_count, coordinate_dim = pos.shape
        if coordinate_dim != self.space_dim:
            raise ValueError(
                f"Expected pos[..., {self.space_dim}], got {tuple(pos.shape)}"
            )
        if field is None:
            if self.in_dim:
                raise ValueError("field is required when in_dim > 0")
            inputs = pos if self.include_pos else pos.new_empty(batch_size, point_count, 0)
        else:
            if field.shape != (batch_size, point_count, self.in_dim):
                raise ValueError(
                    f"Expected field [B, {point_count}, {self.in_dim}], "
                    f"got {tuple(field.shape)}"
                )
            inputs = torch.cat((pos, field), dim=-1) if self.include_pos else field

        feature = self.preprocess_feature(inputs)
        field_feature = self.preprocess_field(inputs) + self.placeholder[None, None, :]
        for block in self.blocks:
            feature, field_feature = block(feature, field_feature)
        return field_feature
