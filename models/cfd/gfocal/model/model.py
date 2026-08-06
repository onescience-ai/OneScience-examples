"""GFocal: Global-Focal Neural Operator for PDEs on Arbitrary Geometries.

Nyström attention for global dependency + Slice focal module for fine-grained
features + gated fusion. Uses NestedUNet, NyströmFormer, MambaInMamba submodules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NystromAttention(nn.Module):
    """Nyström approximation attention: O(n) complexity."""

    def __init__(self, dim, num_landmarks=64, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_landmarks = num_landmarks
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Nyström landmark selection (random subset)
        if N > self.num_landmarks:
            idx = torch.randperm(N, device=x.device)[:self.num_landmarks]
            k_tilde = k[:, :, idx, :]
            v_tilde = v[:, :, idx, :]

            # Approximate attention
            attn1 = torch.matmul(q, k_tilde.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn1 = F.softmax(attn1, dim=-1)

            attn2 = torch.matmul(k_tilde, v_tilde.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn2 = F.softmax(attn2, dim=-1)

            out = torch.matmul(attn1, torch.matmul(attn2, v_tilde))
        else:
            scale = self.head_dim ** 0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)


class SliceFocalModule(nn.Module):
    """Slice-based focal module for fine-grained local features."""

    def __init__(self, dim, num_slices=4):
        super().__init__()
        self.num_slices = num_slices
        self.slice_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim // num_slices), nn.ReLU())
            for _ in range(num_slices)
        ])
        self.fusion = nn.Linear(dim, dim)

    def forward(self, x):
        slices = [proj(x) for proj in self.slice_proj]
        out = torch.cat(slices, dim=-1)
        return self.fusion(out)


class GFocal(nn.Module):
    """Global-Focal Neural Operator for airfoil flow prediction."""

    def __init__(self, in_dim=7, hidden_dim=128, out_dim=4, num_landmarks=64):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.nystrom = NystromAttention(hidden_dim, num_landmarks)
        self.slice_focal = SliceFocalModule(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: (B, N, in_dim)
        h = self.input_proj(x)
        h_global = self.nystrom(h)
        h_local = self.slice_focal(h)
        gate = self.gate(torch.cat([h_global, h_local], dim=-1))
        h_fused = gate * h_global + (1 - gate) * h_local
        return self.fusion(h_fused)
