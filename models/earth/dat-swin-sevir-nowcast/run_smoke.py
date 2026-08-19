"""Minimal smoke test for huilinsigehigh/dat-swin-sevir-nowcast.

This script uses the repository's Swin-T + UNet decoder design, but replaces
SEVIR with a tiny synthetic batch. It performs exactly one forward pass,
computes an MSE loss, runs backward once, and performs one optimizer step.
No pretrained weights and no dataset download are required.
"""

import torch
_tv_lib = torch.library.Library("torchvision", "FRAGMENT")
_tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t


class ChannelAdapter(nn.Module):
    """Collapse T input frames into 3 channels, as in the repo."""
    def __init__(self, in_ch: int, out_ch: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetDecoder(nn.Module):
    def __init__(self, dims=(96, 192, 384, 768), out_ch_top: int = 64):
        super().__init__()
        c1, c2, c3, c4 = dims
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(c1, out_ch_top, kernel_size=4, stride=4),
            nn.GroupNorm(8, out_ch_top),
            nn.SiLU(inplace=True),
        )

    def forward(self, feats):
        f1, f2, f3, f4 = feats
        u3 = self.up3(f4, f3)
        u2 = self.up2(u3, f2)
        u1 = self.up1(u2, f1)
        return self.final_up(u1)


class OfficialSwinEncoder(nn.Module):
    """Same torchvision Swin-T stage split used by train_compare.py."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.adapter = ChannelAdapter(in_ch, 3)
        # weights=None: avoid any network download; smoke test only.
        swin = swin_t(weights=None)
        self.stage1 = nn.Sequential(swin.features[0], swin.features[1])
        self.stage2 = nn.Sequential(swin.features[2], swin.features[3])
        self.stage3 = nn.Sequential(swin.features[4], swin.features[5])
        self.stage4 = nn.Sequential(swin.features[6], swin.features[7])

    def forward(self, x: torch.Tensor):
        x = self.adapter(x)
        h1 = self.stage1(x)
        h2 = self.stage2(h1)
        h3 = self.stage3(h2)
        h4 = self.stage4(h3)
        # torchvision Swin features are NHWC; decoder expects NCHW.
        return [
            h1.permute(0, 3, 1, 2).contiguous(),
            h2.permute(0, 3, 1, 2).contiguous(),
            h3.permute(0, 3, 1, 2).contiguous(),
            h4.permute(0, 3, 1, 2).contiguous(),
        ]


class OfficialSwinNowcast(nn.Module):
    def __init__(self, t_in: int = 6, t_out: int = 6):
        super().__init__()
        self.encoder = OfficialSwinEncoder(t_in)
        self.decoder = UNetDecoder(dims=(96, 192, 384, 768), out_ch_top=64)
        self.head = nn.Conv2d(64, t_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,1,H,W) -> (B,T,H,W)
        if x.dim() == 5 and x.size(2) == 1:
            x = x.squeeze(2)
        feats = self.encoder(x)
        out = self.head(self.decoder(feats))
        return out.unsqueeze(2)


def make_tiny_synthetic_batch(batch_size=1, t_in=6, t_out=6, size=64, device="cpu"):
    """Create a tiny radar-like moving Gaussian blob sequence in [0,1]."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing="ij",
    )
    frames = []
    total = t_in + t_out
    for t in range(total):
        cx = -0.55 + 1.1 * t / max(total - 1, 1)
        cy = -0.25 + 0.5 * t / max(total - 1, 1)
        blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.08)
        frames.append(blob)
    seq = torch.stack(frames, dim=0).unsqueeze(0).unsqueeze(2)
    seq = seq.repeat(batch_size, 1, 1, 1, 1)
    return seq[:, :t_in], seq[:, t_in:]


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    t_in, t_out, size = 6, 6, 64
    x, y = make_tiny_synthetic_batch(1, t_in, t_out, size, device)
    print("synthetic input shape :", tuple(x.shape))
    print("synthetic target shape:", tuple(y.shape))

    model = OfficialSwinNowcast(t_in=t_in, t_out=t_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pred = model(x)
    loss = F.mse_loss(pred, y)
    print("prediction shape:", tuple(pred.shape))
    print("loss before update:", float(loss.detach().cpu()))

    loss.backward()
    optimizer.step()

    print("forward: OK")
    print("backward: OK")
    print("optimizer step: OK")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
