"""
Stormer: A Transformer-based Global Weather Forecasting Model.

Reference:
- "Stormer: A Transformer-based Data-driven Model for Global Weather Forecasting"
- Official repo: https://github.com/microsoft/stormer

This implementation:
- Removes dependency on xformers (replaced with torch.nn.functional.scaled_dot_product_attention)
- Removes dependency on timm (PatchEmbed, Mlp, trunc_normal_ reimplemented)
- Compatible with onescience framework
- Follows official code logic and precision exactly
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
# Constants (fields that should not be predicted — output zero diff)
# ============================================================================

# These are invariant/constant fields in the WeatherBench2 dataset.
# Stormer's 69 atmospheric variables do NOT include any of these,
# but we define them for completeness if the variable set is extended.
CONSTANTS = [
    "anisotropy_of_sub_gridscale_orography",
    "orography",
    "land_sea_mask",
    "slt",
    "lattitude",
    "longitude",
    "angle_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
]


# ============================================================================
# Model metadata for onescience framework
# ============================================================================

@dataclass
class MetaData(ModelMetaData):
    name: str = "Stormer"
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
    """Truncated normal initialization (replaces timm's trunc_normal_).

    Operates on tensor.data to avoid issues with requires_grad=True parameters.
    """
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # Work on .data to avoid in-place operation errors on grad-enabled tensors
    t = tensor.data if hasattr(tensor, 'data') else tensor

    if mean < a - 2 * std or mean > b + 2 * std:
        import warnings
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    t.uniform_(2 * l - 1, 2 * u - 1)
    t.erfinv_()
    t.mul_(std * math.sqrt(2.))
    t.add_(mean)
    t.clamp_(min=a, max=b)


def trunc_normal_(tensor, std=0.02):
    """Drop-in replacement for timm's trunc_normal_."""
    _trunc_normal_(tensor, mean=0., std=std, a=-2., b=2.)


# ============================================================================
# Basic building blocks (replacing timm dependencies)
# ============================================================================

class Mlp(nn.Module):
    """MLP with GELU activation (replaces timm's Mlp)."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding (replaces timm's PatchEmbed).

    Splits image into patches and embeds each patch via Conv2d.
    """
    def __init__(self, patch_size=2, in_chans=1, embed_dim=1024):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=self.patch_size, stride=self.patch_size)
        self.num_patches = None  # set externally after init

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, D, H/p, W/p
        x = x.flatten(2).transpose(1, 2)  # B, L, D
        return x


# ============================================================================
# Position embedding utilities (from official Stormer pos_embed.py)
# ============================================================================

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sine-cosine position embedding from grid positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """2D sine-cosine position embedding."""
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])

    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


# ============================================================================
# Chunked attention (replaces FlashAttention / mem-efficient SDPA)
# ============================================================================

def _chunked_attention(q, k, v, scale, chunk_size=1024):
    """Query-chunked scaled dot-product attention.

    Processes queries in chunks to limit peak memory to O(chunk_size × N)
    instead of O(N × N). This works around:
    - FlashAttention library not being available on DCU/HIP
    - OOM from materializing the full N×N attention matrix

    Args:
        q: (B, num_heads, N, head_dim)
        k: (B, num_heads, N, head_dim)
        v: (B, num_heads, N, head_dim)
        scale: attention scale factor
        chunk_size: number of query tokens per chunk

    Returns:
        (B, num_heads, N, head_dim)
    """
    B, H, N, D = q.shape
    out = torch.empty_like(q)

    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        q_chunk = q[:, :, chunk_start:chunk_end]  # (B, H, chunk, D)

        # Attention scores: (B, H, chunk, N)
        attn = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum: (B, H, chunk, D)
        out[:, :, chunk_start:chunk_end] = torch.matmul(attn, v)

    return out


# ============================================================================
# adaLN-Zero modulation and timestep embedding
# ============================================================================

def modulate(x, shift, scale):
    """Adaptive layer norm modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps (time intervals) into vector representations."""
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Linear(1, hidden_size)

    def forward(self, t):
        return self.mlp(t.unsqueeze(-1))


# ============================================================================
# Memory-efficient attention (replacing xformers)
# ============================================================================

class MemEffAttention(nn.Module):
    """Multi-head attention with memory-efficient chunked implementation.

    Uses query-chunked attention to avoid materializing the full N×N
    attention matrix, working around both FlashAttention library
    unavailability and OOM issues on memory-constrained hardware.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.chunk_size = chunk_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # Replace xformers.ops.unbind with torch.unbind
        q, k, v = torch.unbind(qkv, dim=2)

        # Transpose to (B, num_heads, N, head_dim)
        q = q.permute(0, 2, 1, 3)  # B, num_heads, N, head_dim
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Query-chunked attention: avoids N×N matrix and FlashAttention dep
        x = _chunked_attention(q, k, v, self.scale, self.chunk_size)

        x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================================
# adaLN-Zero Transformer Block
# ============================================================================

class Block(nn.Module):
    """A transformer block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


# ============================================================================
# Final prediction layer
# ============================================================================

class FinalLayer(nn.Module):
    """Final layer with adaLN modulation, maps embeddings to pixel outputs."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ============================================================================
# Weather Embedding: variable tokenization + aggregation
# ============================================================================

class WeatherEmbedding(nn.Module):
    """Variable-specific patch embedding with cross-attention aggregation.

    Each variable gets its own PatchEmbed. Variable tokens are aggregated
    via a learnable query + single-layer cross-attention.
    """

    def __init__(
        self,
        variables,
        img_size,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables

        # Variable tokenization: separate embedding layer for each input variable
        self.token_embeds = nn.ModuleList([
            PatchEmbed(patch_size, 1, embed_dim) for _ in range(len(variables))
        ])
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # Variable embedding to denote which variable each token belongs to
        self.channel_embed, self.channel_map = self._create_var_embedding(embed_dim)

        # Variable aggregation: learnable query + single-layer cross attention
        self.channel_query = nn.Parameter(
            torch.zeros(1, 1, embed_dim), requires_grad=True
        )
        self.channel_agg = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
        )

        self.initialize_weights()

    def _create_var_embedding(self, dim):
        var_embed = nn.Parameter(
            torch.zeros(1, len(self.variables), dim), requires_grad=True
        )
        var_map = {var: idx for idx, var in enumerate(self.variables)}
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(tuple(vars), var_emb.device)
        return var_emb[:, ids, :]

    def initialize_weights(self):
        # Initialize pos_emb and var_emb with sinusoidal encodings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], np.arange(len(self.variables))
        )
        self.channel_embed.data.copy_(
            torch.from_numpy(channel_embed).float().unsqueeze(0)
        )

        # Token embedding layers
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            _trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def aggregate_variables(self, x: torch.Tensor):
        """Aggregate variable tokens via cross-attention.

        Args:
            x: (B, V, L, D)
        Returns:
            (B, L, D)
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # B*L, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(var_query, x, x)  # B*L, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward(self, x: torch.Tensor, variables):
        """Forward pass of weather embedding.

        Args:
            x: (B, V, H, W) input weather state
            variables: list of variable names
        Returns:
            (B, L, D) aggregated token embeddings
        """
        if isinstance(variables, list):
            variables = tuple(variables)

        # Tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            idx = var_ids[i]
            embed_variable = self.token_embeds[idx](x[:, i: i + 1])  # B, L, D
            embeds.append(embed_variable)

        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # Add variable embedding and position embedding
        var_embed = self.get_var_emb(self.channel_embed, list(variables))
        x = x + var_embed.unsqueeze(2)
        x = x + self.pos_embed.unsqueeze(1)

        # Variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        return x


# ============================================================================
# Main Stormer Model
# ============================================================================

class Stormer(nn.Module):
    """Stormer: A Transformer-based Global Weather Forecasting Model.

    This model predicts weather state differences (deltas) over a given
    time interval, conditioned on that interval via adaLN-Zero.

    Args:
        in_img_size (tuple): Input spatial dimensions (H, W).
        variables (list): List of variable name strings.
        patch_size (int): Patch size for tokenization. Default: 2.
        hidden_size (int): Hidden dimension throughout the model. Default: 1024.
        depth (int): Number of transformer blocks. Default: 24.
        num_heads (int): Number of attention heads. Default: 16.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
    """

    def __init__(
        self,
        in_img_size,
        variables,
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()

        # Pad height if not divisible by patch_size
        self.pad_size = 0
        if in_img_size[0] % patch_size != 0:
            self.pad_size = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + self.pad_size, in_img_size[1])

        self.in_img_size = in_img_size
        self.variables = variables
        self.patch_size = patch_size

        # Embedding
        self.embedding = WeatherEmbedding(
            variables=variables,
            img_size=in_img_size,
            patch_size=patch_size,
            embed_dim=hidden_size,
            num_heads=num_heads,
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size)

        # Interval embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Backbone
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Prediction layer
        self.head = FinalLayer(hidden_size, patch_size, len(variables))

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights following official implementation."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer adaLN and linear
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def replace_constant(self, yhat, out_variables):
        """Zero out predicted diffs for constant/invariant variables.

        Following the official Stormer implementation, constant fields
        (like land_sea_mask, orography, etc.) should have zero prediction
        since they don't change over time.

        Args:
            yhat: (B, V, H, W) predicted diffs
            out_variables: list of variable names
        Returns:
            yhat with constant channels set to zero
        """
        for i in range(yhat.shape[1]):
            if out_variables[i] in CONSTANTS:
                yhat[:, i] = 0.0
        return yhat

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """Convert patch tokens back to image space.

        Args:
            x: (B, L, V * patch_size**2)
            h, w: optional height/width override
        Returns:
            imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = len(self.variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1], f"Token count mismatch: {h}*{w} != {x.shape[1]}"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def pad(self, x: torch.Tensor):
        """Pad input height to be divisible by patch_size."""
        h = x.shape[-2]
        if h % self.patch_size != 0:
            pad_size = self.patch_size - h % self.patch_size
            padded_x = F.pad(x, (0, 0, pad_size, 0), 'constant', 0)
        else:
            padded_x = x
            pad_size = 0
        return padded_x, pad_size

    def forward(self, x, variables, time_interval, use_checkpoint=False):
        """Forward pass of Stormer.

        Args:
            x: (B, V, H, W) input weather state (normalized)
            variables: list of variable name strings
            time_interval: (B,) or scalar, time interval in hours, will be divided by 10
            use_checkpoint: if True, apply gradient checkpointing to each block
                (saves memory during training, trades compute for memory)

        Returns:
            (B, V, H_original, W) predicted difference (delta) in normalized space
        """
        # Normalize time interval to [0.6, 1.2, 2.4] range for [6, 12, 24] hours
        if not isinstance(time_interval, torch.Tensor):
            time_interval = torch.tensor([time_interval], device=x.device, dtype=x.dtype)
        time_interval = time_interval / 10.0

        # Pad input height if needed (for patch_size alignment)
        if self.pad_size > 0:
            x = F.pad(x, (0, 0, self.pad_size, 0), 'constant', 0)

        # Embedding (optionally checkpointed — most memory-intensive after blocks)
        if use_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._do_embed, x, variables,
                use_reentrant=False,
            )
        else:
            x = self._do_embed(x, variables)

        # Time interval embedding
        time_interval_emb = self.t_embedder(time_interval)

        # Transformer backbone
        for block in self.blocks:
            if use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, time_interval_emb,
                    use_reentrant=False,
                )
            else:
                x = block(x, time_interval_emb)

        # Prediction head (no checkpoint needed — small)
        x = self.head(x, time_interval_emb)
        x = self.unpatchify(x)

        # Crop back to original height
        if self.pad_size > 0:
            x = x[:, :, self.pad_size:]

        return x

    def _do_embed(self, x, variables):
        """Embedding step (extracted for checkpointing)."""
        x = self.embedding(x, variables)  # B, L, D
        x = self.embed_norm_layer(x)
        return x
