# -*- coding: utf-8 -*-
"""
vqvae_torch.py —— diffusers.VQModel 的 **纯 PyTorch 等价实现**(备用后端 / Plan B)
================================================================================
为什么需要它
--------------------------------------------------------------------------------
flagos_earth_onecode 镜像里 `import diffusers` 的依赖链会连带触发某个已损坏的
动态库(libtensorflow_cc.so.2: undefined symbol: ncclCommRegister),导致
`from diffusers import VQModel` 直接 ImportError。该问题与模型本身无关,属于镜像
的第三方库链接问题。

本文件按 diffusers 0.17/0.30 中 `VQModel` 的**结构与参数命名**逐层复刻,只依赖
torch,不 import diffusers / transformers / tensorflow。它:

  * 用 `config.json` 里的超参构建完全同构的网络;
  * 直接读取官方权重 `diffusion_pytorch_model.bin`;
  * 以 **strict=True** 装载 —— 只要有任何一个参数名或形状对不上就会报错退出,
    因此"能加载成功"本身即是结构等价的强校验,不存在蒙混过关的可能;
  * 暴露与 diffusers 相同的 API:`model.encode(x).latents` / `model(x).sample`。

自检用法(可单独运行):
    python vqvae_torch.py --inspect     # 打印权重文件里的全部参数名与形状
    python vqvae_torch.py               # 构建 + strict 加载 + 跑一次前向
"""

import os
import json
import math
import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 基础模块(命名严格对齐 diffusers,以保证 state_dict 的 key 一致)
# ==============================================================================
class ResnetBlock2D(nn.Module):
    """对应 diffusers.models.resnet.ResnetBlock2D(temb_channels=None, pre_norm=True)"""

    def __init__(self, in_channels, out_channels, groups=8, eps=1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            if in_channels != out_channels else None
        )

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h            # output_scale_factor = 1.0


class Attention(nn.Module):
    """对应 diffusers.models.attention_processor.Attention
    (VAE 里的自注意力:heads=1, dim_head=channels, residual_connection=True,
     rescale_output_factor=1.0, upcast_softmax=True)"""

    def __init__(self, channels, groups=8, eps=1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels, bias=True), nn.Dropout(0.0)])
        self.scale = channels ** -0.5      # dim_head ** -0.5,dim_head = channels(单头)

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape
        hs = x.view(b, c, h * w).transpose(1, 2)                 # (B, HW, C)
        hs = self.group_norm(hs.transpose(1, 2)).transpose(1, 2)  # GroupNorm 作用在通道维
        q, k, v = self.to_q(hs), self.to_k(hs), self.to_v(hs)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.float().softmax(dim=-1).to(q.dtype)           # upcast_softmax
        hs = torch.matmul(attn, v)
        hs = self.to_out[0](hs)
        hs = self.to_out[1](hs)
        hs = hs.transpose(-1, -2).reshape(b, c, h, w)
        return hs + residual                                      # residual_connection, rescale=1.0


class Downsample2D(nn.Module):
    """对应 diffusers Downsample2D(use_conv=True, padding=0, name='op')"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample2D(nn.Module):
    """对应 diffusers Upsample2D(use_conv=True)"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, groups=8, add_downsample=True, attn=False):
        super().__init__()
        resnets, attentions = [], []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(cin, out_channels, groups=groups))
            if attn:
                attentions.append(Attention(out_channels, groups=groups))
        self.resnets = nn.ModuleList(resnets)
        if attn:
            self.attentions = nn.ModuleList(attentions)
        else:
            self.attentions = None
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels)]) if add_downsample else None

    def forward(self, x):
        for i, resnet in enumerate(self.resnets):
            x = resnet(x)
            if self.attentions is not None:
                x = self.attentions[i](x)
        if self.downsamplers is not None:
            for d in self.downsamplers:
                x = d(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, groups=8, add_upsample=True, attn=False):
        super().__init__()
        resnets, attentions = [], []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(cin, out_channels, groups=groups))
            if attn:
                attentions.append(Attention(out_channels, groups=groups))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions) if attn else None
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None

    def forward(self, x):
        for i, resnet in enumerate(self.resnets):
            x = resnet(x)
            if self.attentions is not None:
                x = self.attentions[i](x)
        if self.upsamplers is not None:
            for u in self.upsamplers:
                x = u(x)
        return x


class UNetMidBlock2D(nn.Module):
    """resnets[0] -> attentions[0] -> resnets[1]"""

    def __init__(self, channels, groups=8):
        super().__init__()
        self.attentions = nn.ModuleList([Attention(channels, groups=groups)])
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels, groups=groups),
            ResnetBlock2D(channels, channels, groups=groups),
        ])

    def forward(self, x):
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x


# ==============================================================================
# Encoder / Decoder
# ==============================================================================
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, down_block_types, block_out_channels,
                 layers_per_block, norm_num_groups):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        blocks = []
        output_channel = block_out_channels[0]
        for i, btype in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final = (i == len(block_out_channels) - 1)
            blocks.append(DownEncoderBlock2D(
                input_channel, output_channel,
                num_layers=layers_per_block, groups=norm_num_groups,
                add_downsample=not is_final,
                attn=btype.startswith("Attn"),
            ))
        self.down_blocks = nn.ModuleList(blocks)
        self.mid_block = UNetMidBlock2D(block_out_channels[-1], groups=norm_num_groups)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1],
                                          num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        return self.conv_out(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, up_block_types, block_out_channels,
                 layers_per_block, norm_num_groups):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.mid_block = UNetMidBlock2D(block_out_channels[-1], groups=norm_num_groups)
        reversed_channels = list(reversed(block_out_channels))
        blocks = []
        output_channel = reversed_channels[0]
        for i, btype in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_channels[i]
            is_final = (i == len(block_out_channels) - 1)
            blocks.append(UpDecoderBlock2D(
                prev_output_channel, output_channel,
                num_layers=layers_per_block + 1, groups=norm_num_groups,
                add_upsample=not is_final,
                attn=btype.startswith("Attn"),
            ))
        self.up_blocks = nn.ModuleList(blocks)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0],
                                          num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.conv_in(z)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        return self.conv_out(x)


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, vq_embed_dim):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.embedding = nn.Embedding(n_e, vq_embed_dim)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.vq_embed_dim)
        try:
            dist = torch.cdist(z_flat, self.embedding.weight)
        except Exception:
            # 某些算子库上 cdist 可能不受支持,用等价的展开式
            dist = (z_flat.pow(2).sum(1, keepdim=True)
                    + self.embedding.weight.pow(2).sum(1)
                    - 2 * z_flat @ self.embedding.weight.t()).clamp_min(0).sqrt()
        idx = torch.argmin(dist, dim=1)
        z_q = self.embedding(idx).view(z.shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, idx


# ==============================================================================
# VQModel(与 diffusers 同名 API)
# ==============================================================================
class _Out:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getitem__(self, i):
        return list(self.__dict__.values())[i]


class VQModelLite(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.config = SimpleNamespace(**cfg)
        boc = list(cfg["block_out_channels"])
        groups = int(cfg.get("norm_num_groups", 32))
        latent_channels = int(cfg.get("latent_channels", 1))
        vq_embed_dim = int(cfg.get("vq_embed_dim") or latent_channels)

        self.encoder = Encoder(
            in_channels=int(cfg["in_channels"]),
            out_channels=latent_channels,
            down_block_types=cfg["down_block_types"],
            block_out_channels=boc,
            layers_per_block=int(cfg.get("layers_per_block", 1)),
            norm_num_groups=groups,
        )
        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, kernel_size=1)
        self.quantize = VectorQuantizer(int(cfg["num_vq_embeddings"]), vq_embed_dim)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, kernel_size=1)
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=int(cfg["out_channels"]),
            up_block_types=cfg["up_block_types"],
            block_out_channels=boc,
            layers_per_block=int(cfg.get("layers_per_block", 1)),
            norm_num_groups=groups,
        )

    # --- 与 diffusers 一致的 API ---
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return _Out(latents=h)

    def decode(self, h):
        quant, _ = self.quantize(h)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return _Out(sample=dec)

    def forward(self, sample):
        h = self.encode(sample).latents
        return _Out(sample=self.decode(h).sample)


# ==============================================================================
# 权重加载
# ==============================================================================
_LEGACY_ATTN_MAP = {
    ".query.": ".to_q.",
    ".key.": ".to_k.",
    ".value.": ".to_v.",
    ".proj_attn.": ".to_out.0.",
}


def _remap_legacy_attention(state_dict):
    """diffusers<=0.17 的 VAE 用 AttentionBlock,参数名是 query/key/value/proj_attn;
    0.18+ 改名为 to_q/to_k/to_v/to_out.0。此处做等价重命名(仅改名,不改形状)。"""
    out, n = {}, 0
    for k, v in state_dict.items():
        nk = k
        for old, new in _LEGACY_ATTN_MAP.items():
            if old in nk:
                nk = nk.replace(old, new)
                n += 1
                break
        out[nk] = v
    return out, n


def load_state_dict_file(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


def find_weight_file(model_dir):
    for name in ("diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{model_dir} 下未找到 diffusion_pytorch_model.bin/.safetensors")


def from_pretrained(model_dir, verbose=True):
    """构建 VQModelLite 并以 strict=True 装载官方权重。任何 key/shape 不一致都会抛错。"""
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = VQModelLite(cfg)

    wpath = find_weight_file(model_dir)
    sd = load_state_dict_file(wpath)
    sd, n_renamed = _remap_legacy_attention(sd)
    if verbose and n_renamed:
        print(f"[fallback] 检测到旧版注意力参数命名,已等价重命名 {n_renamed} 项 "
              f"(query/key/value/proj_attn -> to_q/to_k/to_v/to_out.0)")

    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        own = set(model.state_dict().keys())
        got = set(sd.keys())
        missing = sorted(own - got)
        unexpected = sorted(got - own)
        print("\n[fallback][ERROR] strict 加载失败,结构与权重不一致。详情如下:")
        print(f"  权重文件      : {wpath}")
        print(f"  缺失参数({len(missing)}):")
        for k in missing[:40]:
            print(f"    - {k}  {tuple(model.state_dict()[k].shape)}")
        print(f"  多余参数({len(unexpected)}):")
        for k in unexpected[:40]:
            print(f"    + {k}  {tuple(sd[k].shape)}")
        print("\n  原始报错:")
        print("  " + str(e)[:2000])
        raise

    if verbose:
        print(f"[fallback] 已用纯 PyTorch 等价实现装载官方权重(strict=True 校验通过): {wpath}")
    return model


# ==============================================================================
# 自检入口
# ==============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--inspect", action="store_true", help="只打印权重文件里的参数名与形状")
    a = ap.parse_args()

    if a.inspect:
        wpath = find_weight_file(a.model_dir)
        sd = load_state_dict_file(wpath)
        print(f"权重文件: {wpath}   参数张量数: {len(sd)}")
        total = 0
        for k, v in sd.items():
            total += v.numel()
            print(f"  {k:<70} {tuple(v.shape)}")
        print(f"参数总量: {total:,}")
        raise SystemExit(0)

    m = from_pretrained(a.model_dir).eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"参数总量: {n:,}")
    x = torch.randn(1, int(m.config.in_channels),
                    int(m.config.sample_size[0]), int(m.config.sample_size[1]))
    with torch.no_grad():
        lat = m.encode(x).latents
        rec = m(x).sample
    print(f"input {tuple(x.shape)} -> latents {tuple(lat.shape)} -> recon {tuple(rec.shape)}")
