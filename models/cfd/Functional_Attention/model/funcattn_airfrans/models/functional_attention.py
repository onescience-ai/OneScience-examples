"""PyTorch implementation of Functional Attention."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - depends on runtime env.
    raise RuntimeError("PyTorch is required for funcattn_repro.model") from exc


class FunctionalAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        heads: int = 8,
        bases: int = 32,
        ridge_lambda: float = 1e-3,
        dropout: float = 0.0,
        basis_q: nn.Linear | None = None,
        basis_kv: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.bases = bases
        self.ridge_lambda = ridge_lambda
        self.qkv = nn.Linear(channels, channels * 3)
        self.basis_q = basis_q if basis_q is not None else nn.Linear(channels, heads * bases)
        self.basis_kv = basis_kv if basis_kv is not None else nn.Linear(channels, heads * bases)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, num_points, _channels = x.shape
        qkv = self.qkv(x).view(bsz, num_points, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        phi = self.basis_q(x).view(bsz, num_points, self.heads, self.bases)
        psi = self.basis_kv(x).view(bsz, num_points, self.heads, self.bases)
        phi = phi.permute(0, 2, 1, 3).softmax(dim=-1)
        psi = psi.permute(0, 2, 1, 3).softmax(dim=-1)

        q_tilde = torch.einsum("bhnk,bhnd->bhkd", phi, q)
        k_tilde = torch.einsum("bhnk,bhnd->bhkd", psi, k)
        v_tilde = torch.einsum("bhnk,bhnd->bhkd", psi, v)

        kk = k_tilde @ k_tilde.transpose(-1, -2)
        qk = q_tilde @ k_tilde.transpose(-1, -2)
        eye = torch.eye(self.bases, dtype=x.dtype, device=x.device)
        system = kk + self.ridge_lambda * eye
        operator = torch.linalg.solve(system, qk.transpose(-1, -2)).transpose(-1, -2)

        transported = operator @ v_tilde
        out = torch.einsum("bhnk,bhkd->bhnd", phi, transported)
        out = out.transpose(1, 2).reshape(bsz, num_points, self.channels)
        return self.proj(self.dropout(out))


class FuncAttnBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        bases: int,
        ridge_lambda: float,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
        basis_q: nn.Linear | None = None,
        basis_kv: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = FunctionalAttention(
            channels,
            heads=heads,
            bases=bases,
            ridge_lambda=ridge_lambda,
            dropout=dropout,
            basis_q=basis_q,
            basis_kv=basis_kv,
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FunctionalAttentionRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        *,
        channels: int = 256,
        layers: int = 8,
        heads: int = 8,
        bases: int = 32,
        ffn_ratio: int = 4,
        ridge_lambda: float = 1e-3,
        dropout: float = 0.0,
        share_basis: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, channels), nn.GELU())
        shared_basis_q = nn.Linear(channels, heads * bases) if share_basis else None
        shared_basis_kv = nn.Linear(channels, heads * bases) if share_basis else None
        self.blocks = nn.ModuleList(
            [
                FuncAttnBlock(
                    channels,
                    heads=heads,
                    bases=bases,
                    ridge_lambda=ridge_lambda,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    basis_q=shared_basis_q,
                    basis_kv=shared_basis_kv,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, output_dim)

    def forward(self, x, mask=None):
        hidden = self.encoder(x)
        for block in self.blocks:
            hidden = block(hidden)
        pred = self.head(self.norm(hidden))
        if mask is not None:
            pred = pred * mask.unsqueeze(-1).to(pred.dtype)
        return pred
