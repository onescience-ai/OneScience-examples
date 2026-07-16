from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


class StandardMLP(nn.Module):
    """Pointwise MLP with state-dict names compatible with OneScience."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "gelu",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        dimensions = [int(input_dim), *(int(value) for value in hidden_dims), int(output_dim)]
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=use_bias)
            for in_features, out_features in zip(dimensions[:-1], dimensions[1:])
        )
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class OneMlp(nn.Module):
    """Small compatibility wrapper used by released OneScience checkpoints."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mlp = StandardMLP(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _PoolingReducer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_in(x)
        if x.ndim > 3:
            x = x.mean(dim=tuple(range(2, x.ndim - 1)))
        return self.out_ffn(x)


class _SwapGridAxes(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1, 3)


class _FactAttnWeight(nn.Module):
    def __init__(self, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        self.dim_head = int(dim_head)
        self.heads = int(heads)
        self.scale = self.dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        x = x.reshape(batch, length, self.heads, self.dim_head)
        x = x.permute(0, 2, 1, 3).contiguous()
        query = self.to_q(x)
        key = self.to_k(x)
        return self.softmax(torch.matmul(query, key.transpose(-1, -2)) * self.scale)


class FactAttention2D(nn.Module):
    """Factorized axial attention for a structured two-dimensional grid."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        shapelist: Sequence[int],
    ) -> None:
        super().__init__()
        if len(shapelist) != 2:
            raise ValueError("FactAttention2D requires shapelist=(height, width)")
        if dim != heads * dim_head:
            raise ValueError("dim must equal heads * dim_head")
        self.dim_head = int(dim_head)
        self.heads = int(heads)
        self.H, self.W = (int(value) for value in shapelist)
        self.attn_x = _FactAttnWeight(self.heads, self.dim_head, dropout)
        self.attn_y = _FactAttnWeight(self.heads, self.dim_head, dropout)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_x = nn.Sequential(_PoolingReducer(dim, dim, dim))
        self.to_y = nn.Sequential(_SwapGridAxes(), _PoolingReducer(dim, dim, dim))
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        if length != self.H * self.W:
            raise ValueError(
                f"Sequence length {length} does not match grid {self.H}x{self.W}"
            )
        grid = x.reshape(batch, self.H, self.W, channels).contiguous()
        values = self.to_v(
            grid.reshape(batch, self.H, self.W, self.heads, self.dim_head)
        )
        values = values.permute(0, 3, 1, 2, 4).contiguous()
        result_x = torch.einsum(
            "bhij,bhjmc->bhimc", self.attn_x(self.to_x(grid)), values
        )
        result_y = torch.einsum(
            "bhlm,bhimc->bhilc", self.attn_y(self.to_y(grid)), result_x
        )
        result = result_y.permute(0, 2, 3, 1, 4).contiguous()
        result = result.reshape(batch, length, channels)
        return self.to_out(result)


class OneAttention(nn.Module):
    """Compatibility wrapper retaining the original checkpoint hierarchy."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.attentioner = FactAttention2D(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attentioner(x)


class FactformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
        mlp_ratio: int,
        spatial_shape: Sequence[int],
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = OneAttention(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            shapelist=spatial_shape,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = OneMlp(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            hidden_dims=[hidden_dim * mlp_ratio],
            activation=activation,
            use_bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Attn(self.ln_1(x)) + x
        return self.mlp(self.ln_2(x)) + x


class OneTransformer(nn.Module):
    """Compatibility wrapper retaining the original checkpoint hierarchy."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.transformer = FactformerBlock(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


class FactFormer2D(nn.Module):
    """FactFormer neural operator for structured 2D flow fields.

    ``pos`` has shape ``(B, H*W, 2)`` and ``fx`` has shape
    ``(B, H*W, t_in*out_dim)``. The output contains ``latent_steps`` future
    fields concatenated along the final dimension.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spatial_shape: Sequence[int],
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        include_pos: bool = True,
        space_dim: int = 2,
        latent_multiplier: float = 2.0,
        max_latent_steps: int = 4,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.spatial_shape = tuple(int(value) for value in spatial_shape)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.include_pos = bool(include_pos)
        self.space_dim = int(space_dim)
        self.latent_dim = int(self.hidden_dim * float(latent_multiplier))
        self.max_latent_steps = int(max_latent_steps)

        if len(self.spatial_shape) != 2:
            raise ValueError("FactFormer2D requires a two-dimensional spatial shape")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.max_latent_steps < 1:
            raise ValueError("max_latent_steps must be positive")

        input_dim = self.in_dim + (self.space_dim if self.include_pos else 0)
        self.preprocess = OneMlp(
            input_dim=input_dim,
            output_dim=self.hidden_dim,
            hidden_dims=[self.hidden_dim * 2],
            activation=activation,
            use_bias=True,
        )
        self.blocks = nn.ModuleList(
            OneTransformer(
                num_heads=self.heads,
                hidden_dim=self.hidden_dim,
                dropout=float(dropout),
                activation=activation,
                mlp_ratio=int(mlp_ratio),
                spatial_shape=self.spatial_shape,
            )
            for _ in range(self.depth)
        )
        self.expand_latent = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
        self.latent_time_embedding = nn.Parameter(
            torch.randn(1, self.max_latent_steps, 1, self.latent_dim) * 0.02
        )
        self.propagator = OneMlp(
            input_dim=self.latent_dim,
            output_dim=self.latent_dim,
            hidden_dims=[self.hidden_dim],
            activation=activation,
            use_bias=True,
        )
        self.to_out = OneMlp(
            input_dim=self.latent_dim,
            output_dim=self.out_dim,
            hidden_dims=[self.hidden_dim],
            activation=activation,
            use_bias=True,
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        pos: torch.Tensor,
        fx: torch.Tensor,
        latent_steps: int = 1,
    ) -> torch.Tensor:
        if fx.ndim != 3:
            raise ValueError(f"fx must have shape (B, N, C), got {tuple(fx.shape)}")
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)
        if pos.shape[0] == 1 and fx.shape[0] > 1:
            pos = pos.expand(fx.shape[0], -1, -1)
        if pos.shape[:2] != fx.shape[:2]:
            raise ValueError(f"Incompatible pos and fx shapes: {pos.shape}, {fx.shape}")
        if not 1 <= int(latent_steps) <= self.max_latent_steps:
            raise ValueError(
                f"latent_steps must be in [1, {self.max_latent_steps}], got {latent_steps}"
            )

        hidden = torch.cat((pos, fx), dim=-1) if self.include_pos else fx
        hidden = self.preprocess(hidden)
        for block in self.blocks:
            hidden = block(hidden)

        latent = self.expand_latent(hidden)
        outputs = []
        for step in range(int(latent_steps)):
            latent = latent + self.latent_time_embedding[:, step]
            latent = self.propagator(latent) + latent
            outputs.append(self.to_out(latent))
        return torch.cat(outputs, dim=-1)


__all__ = ["FactFormer2D"]
