"""Deterministic fake cubed-sphere fields; not a substitute for ERA5."""
import torch


def make_fake_batch(batch=2, channels=2, faces=6, height=8, width=8, seed=7):
    if faces != 6:
        raise ValueError("cubed sphere requires six faces")
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
    continuous = (xx + 0.5 * yy).expand(batch, 1, faces, height, width)
    face_id = torch.arange(faces).view(1, 1, faces, 1, 1).float().expand(batch, 1, faces, height, width)
    bank = torch.cat((continuous, face_id, torch.ones_like(continuous)), 1)
    if channels > bank.shape[1]:
        bank = torch.cat((bank, torch.zeros(batch, channels - bank.shape[1], faces, height, width)), 1)
    x = bank[:, :channels]
    return x + 0.01 * torch.randn(x.shape, generator=g)
