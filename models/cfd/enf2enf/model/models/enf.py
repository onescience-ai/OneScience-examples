"""ENF（Equivariant Neural Field）核心模块（Section 3.2 Eq.4-6）。

包含：
- ENFAttentionHead：单头 equivariant cross attention
- ENFDecoderLayer / ENFEncoder：组合多头注意力输出
- LatentSelfAttention：decoder latent 点间 self-attention（N blocks，residual+LN）

关键公式：
att_{m,j} = exp(q(b)^T k(c)/sqrt(dk) - sigma_w ||x_m - p_j||^2) / sum_l(...)
v(b,c) = (W_v c) .* (W_s b) + W_b b
f(x_m) = W_o sum_j att_{m,j} v(b_{j,m}, c_j)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rff import RandomFourierFeatures


class ENFAttentionHead(nn.Module):
    """单头等变交叉注意力：query 点 x 对 latent 点 {(p_j, c_j)} 的注意力。"""

    def __init__(self, d_gamma: int, l_dim: int, d_out: int, sigma_window: float = 0.1):
        super().__init__()
        dk = d_gamma
        self.w_q = nn.Linear(d_gamma, dk, bias=False)
        self.w_k = nn.Linear(l_dim, dk, bias=False)
        self.w_v = nn.Linear(l_dim, d_out, bias=False)
        self.w_s = nn.Linear(d_gamma, d_out, bias=False)
        self.w_b = nn.Linear(d_gamma, d_out, bias=False)
        self.sigma_window = sigma_window

    def forward(self, x: torch.Tensor, p: torch.Tensor, c: torch.Tensor, rff: RandomFourierFeatures):
        """x: (B, N, d_in), p: (n_lat, d_in), c: (B, n_lat, l_dim) -> (B, N, d_out)"""
        B, N, _ = x.shape
        n_lat = p.shape[0]
        # offsets b_{j,m} = gamma(x_m - p_j)
        diff = x.unsqueeze(2) - p.unsqueeze(0).unsqueeze(0)  # (B,N,n_lat,d_in)
        b = rff(diff)  # (B,N,n_lat,d_gamma)
        q = self.w_q(b)  # (B,N,n_lat,dk)
        k = self.w_k(c)  # (B,n_lat,dk) -> unsqueeze(1)
        k = k.unsqueeze(1)  # (B,1,n_lat,dk)
        dk = q.shape[-1]
        att_logits = (q * k).sum(-1) / (dk ** 0.5)  # (B,N,n_lat)
        # Gaussian window: -sigma ||x_m - p_j||^2
        dist2 = diff.pow(2).sum(-1)  # (B,N,n_lat)
        att_logits = att_logits - self.sigma_window * dist2
        att = F.softmax(att_logits, dim=-1)  # (B,N,n_lat)
        v = (self.w_v(c).unsqueeze(1) * self.w_s(b) + self.w_b(b))  # (B,1,n_lat,d_out) * (B,N,n_lat,d_out)
        # att (B,N,n_lat) -> (B,N,n_lat,1)
        out = (att.unsqueeze(-1) * v).sum(dim=2)  # (B,N,d_out)
        return out


class ENFMultiHead(nn.Module):
    """多头等变交叉注意力，输出拼接后线性混合。"""

    def __init__(self, d_gamma: int, l_dim: int, width: int, heads: int, sigma_window: float):
        super().__init__()
        self.heads = heads
        d_out = width // heads
        self.heads_list = nn.ModuleList(
            [ENFAttentionHead(d_gamma, l_dim, d_out, sigma_window) for _ in range(heads)]
        )
        self.out_proj = nn.Linear(width, width)

    def forward(self, x, p, c, rff):
        hs = [h(x, p, c, rff) for h in self.heads_list]
        out = torch.cat(hs, dim=-1)  # (B,N,width)
        return self.out_proj(out)


class LatentSelfAttentionBlock(nn.Module):
    """latent 点间 self-attention（residual + LayerNorm）。"""

    def __init__(self, l_dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(l_dim)
        self.attn = nn.MultiheadAttention(l_dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(l_dim)
        self.mlp = nn.Sequential(
            nn.Linear(l_dim, 4 * l_dim),
            nn.GELU(),
            nn.Linear(4 * l_dim, l_dim),
        )

    def forward(self, c: torch.Tensor):
        # c: (B, n_lat, l_dim)
        c2 = self.norm1(c)
        a, _ = self.attn(c2, c2, c2)
        c = c + a
        c = c + self.mlp(self.norm2(c))
        return c


class LatentSelfAttention(nn.Module):
    """N 层 latent self-attention 栈。"""

    def __init__(self, l_dim: int, heads: int, blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([LatentSelfAttentionBlock(l_dim, heads) for _ in range(blocks)])

    def forward(self, c: torch.Tensor):
        for blk in self.blocks:
            c = blk(c)
        return c


class ENFEncoder(nn.Module):
    """输入几何编码器 f_theta_a：等价交叉注意力重建几何场。

    输入查询坐标 x 与 latent 点 {(p_j, c_j)}，输出几何场重建 a_hat(x)。
    width=128, heads=2, RFF(128,2)（elasticity）。
    """

    def __init__(
        self,
        coord_dim: int = 2,
        width: int = 128,
        heads: int = 2,
        n_lat: int = 9,
        l_dim: int = 8,
        sigma_window: float = 0.1,
        rff_d: int = 128,
        rff_sigma: float = 2,
        out_channels: int = 2,
        train_rff: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.n_lat = n_lat
        self.l_dim = l_dim
        self.rff = RandomFourierFeatures(rff_d, rff_sigma, coord_dim, trainable=train_rff, seed=seed)
        d_gamma = self.rff.out_dim
        self.attn = ENFMultiHead(d_gamma, l_dim, width, heads, sigma_window)
        self.head = nn.Linear(width, out_channels)

    def forward(self, x: torch.Tensor, p: torch.Tensor, c: torch.Tensor):
        """x:(B,N,2) p:(n_lat,2) c:(B,n_lat,l_dim) -> (B,N,out_channels)"""
        h = self.attn(x, p, c, self.rff)
        return self.head(h)


class ENFDecoder(nn.Module):
    """输出物理场解码器 f_theta_u：latent self-attention + 等价交叉注意力。

    width=256, heads=2, RFF(256,10), attention_blocks=2。
    latent 输入 z_u = {(p_j, c_j)}，其中 c_j 已与全局参数 mu 拼接（elasticity 无 mu）。
    """

    def __init__(
        self,
        coord_dim: int = 2,
        width: int = 256,
        heads: int = 2,
        n_lat: int = 9,
        l_dim: int = 8,
        sigma_window: float = 0.1,
        rff_d: int = 256,
        rff_sigma: float = 10,
        attention_blocks: int = 2,
        out_channels: int = 1,
        train_rff: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.n_lat = n_lat
        self.l_dim = l_dim
        self.rff = RandomFourierFeatures(rff_d, rff_sigma, coord_dim, trainable=train_rff, seed=seed)
        self.self_attn = LatentSelfAttention(l_dim, heads, attention_blocks)
        d_gamma = self.rff.out_dim
        self.attn = ENFMultiHead(d_gamma, l_dim, width, heads, sigma_window)
        self.head = nn.Linear(width, out_channels)

    def forward(self, x: torch.Tensor, p: torch.Tensor, c: torch.Tensor):
        """x:(B,N,2) p:(n_lat,2) c:(B,n_lat,l_dim) -> (B,N,out_channels)"""
        c_tilde = self.self_attn(c)
        h = self.attn(x, p, c_tilde, self.rff)
        return self.head(h)
