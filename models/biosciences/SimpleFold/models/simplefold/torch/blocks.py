#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
from torch import nn
from .layers import modulate, SwiGLUFeedForward

try:
    from timm.models.vision_transformer import Mlp
except ModuleNotFoundError:
    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.drop1 = nn.Dropout(drop)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop2 = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT encoder blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        latents,
        c,
        **kwargs,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        _latents = self.attn(
            modulate(self.norm1(latents), shift_msa, scale_msa), **kwargs
        )
        latents = latents + gate_msa.unsqueeze(1) * _latents
        latents = latents + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(latents), shift_mlp, scale_mlp)
        )
        return latents


class TransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

    def forward(
        self,
        latents,
        **kwargs,
    ):
        _latents = self.attn(self.norm1(latents), **kwargs)
        latents = latents + _latents
        latents = latents + self.mlp(self.norm2(latents))
        return latents


class HomogenTrunk(nn.Module):
    def __init__(self, block, depth):
        super().__init__()
        self.blocks = nn.ModuleList([block() for _ in range(depth)])

    def forward(self, latents, c, **kwargs):
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)
        return latents
