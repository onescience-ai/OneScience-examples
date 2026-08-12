"""
ClimaX: A foundation model for weather and climate.

Reference:
- "ClimaX: A foundation model for weather and climate" (arXiv:2301.10343)
- Official repo: https://github.com/microsoft/ClimaX

This implementation:
- Removes dependency on timm (PatchEmbed, Block, trunc_normal_ reimplemented)
- Removes dependency on pytorch_lightning
- Compatible with onescience framework
- Follows official code logic and precision exactly
- Supports training from scratch without pretrained weights
"""

import math
import numpy as np
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.models.meta import ModelMetaData


# ============================================================================
# Model metadata for onescience framework
# ============================================================================

@dataclass
class MetaData(ModelMetaData):
    name: str = "ClimaX"
    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    amp_cpu: bool = None
    amp_gpu: bool = None
    onnx_cpu: bool = False
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


# ============================================================================
# Utility functions (replacing timm dependencies)
# ============================================================================

def _trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization (replaces timm's trunc_normal_)."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)


def trunc_normal_(tensor, std=0.02):
    """Drop-in replacement for timm's trunc_normal_ (wrapped with no_grad)."""
    with torch.no_grad():
        _trunc_normal_(tensor, mean=0., std=std, a=-2., b=2.)


# ============================================================================
# Position embedding utilities (from official ClimaX pos_embed.py)
# ============================================================================

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size_h: int of the grid height
    grid_size_w: int of the grid width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


# ============================================================================
# Basic building blocks (replacing timm dependencies)
# ============================================================================

class Mlp(nn.Module):
    """MLP with GELU activation (replaces timm.layers.Mlp)."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (replaces timm's DropPath)."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding (replaces timm's PatchEmbed).

    Splits image into patches and embeds each patch via Conv2d.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, D, H/p, W/p
        x = x.flatten(2).transpose(1, 2)  # B, L, D
        return x


class Attention(nn.Module):
    """Multi-head self-attention (replaces timm's Attention)."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)  # B, num_heads, N, head_dim
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.attn_drop.p > 0.0 else 0.0,
            scale=self.scale,
        )

        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ViT Block with LayerNorm, Attention, and MLP (replaces timm's Block)."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# Main ClimaX Model
# ============================================================================

class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    This is the base ClimaX architecture for global weather forecasting.
    It uses per-variable tokenization, cross-attention variable aggregation,
    a ViT backbone, and an MLP prediction head.

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data [H, W]
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        decoder_depth (int): number of decoder layers
        num_heads (int): number of attention heads
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
    """

    def __init__(
        self,
        default_vars,
        img_size=(32, 64),
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
    ):
        super().__init__()

        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.default_vars = default_vars

        # variable tokenization: separate embedding layer for each input variable
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1, embed_dim) for _ in range(len(default_vars))]
        )
        self.num_patches = self.token_embeds[0].num_patches

        # variable embedding to denote which variable each token belongs to
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding and lead time embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # ------------------------------------------------------------------
        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ------------------------------------------------------------------
        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, len(self.default_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)

        # ------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb with sinusoidal encodings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        # token embedding layers
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(
            torch.zeros(1, len(self.default_vars), dim), requires_grad=True
        )
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        """
        Aggregate variable tokens via cross-attention.

        Args:
            x: (B, V, L, D)

        Returns:
            (B, L, D)
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)  # BxL, 1, D
        x = x.squeeze(dim=1)  # BxL, D

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        """Encode input weather state into transformer tokens.

        Args:
            x: [B, V, H, W] input climate variables
            lead_times: [B] forecasting lead times (hours, normalized by dividing by 100)
            variables: tuple of input variable names

        Returns:
            [B, L, D] encoded token representations
        """
        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x, variables, out_variables=None, lead_time=None):
        """Forward pass through ClimaX.

        This is the onescience-compatible forward that takes input tensor
        and variable lists, returning predicted weather state.

        Args:
            x: [B, V_in, H, W] input weather/climate variables
            variables: list of input variable name strings
            out_variables: list of output variable name strings (if None, uses all default_vars)
            lead_time: scalar or [B] tensor, forecasting lead time in normalized hours
                       (raw_hours / 100). If None, defaults to 0.72 (72 hours).

        Returns:
            preds: [B, V_out, H, W] predicted weather/climate variables
        """
        if out_variables is None:
            out_variables = self.default_vars

        if lead_time is None:
            lead_time = 0.72  # default 72 hours / 100

        if not isinstance(lead_time, torch.Tensor):
            lead_time = torch.full(
                (x.shape[0],), lead_time,
                device=x.device, dtype=x.dtype
            )

        # Encode
        out_transformers = self.forward_encoder(x, lead_time, variables)  # B, L, D

        # Decode
        preds = self.head(out_transformers)  # B, L, V*p*p
        preds = self.unpatchify(preds)  # B, V_all, H, W

        # Select output variables
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        return preds
