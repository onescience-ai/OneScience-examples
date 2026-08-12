"""Deterministic fake dataset for the tiny AtmoRep training pipeline."""

import torch
from torch.utils.data import Dataset

from .tiny_atmorep import TinyAtmoRepConfig


class FakeAtmoRepDataset(Dataset):
    def __init__(self, config: TinyAtmoRepConfig, samples: int, seed: int, mask_fraction: float) -> None:
        self.config, self.samples, self.seed = config, samples, seed
        self.num_tokens = (config.input_shape[0] // config.patch_shape[0]) * (config.input_shape[2] // config.patch_shape[1]) * (config.input_shape[3] // config.patch_shape[2])
        self.mask_count = max(1, round(self.num_tokens * mask_fraction))

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        fields = torch.randn(self.config.input_shape, generator=generator)
        mask = torch.zeros(self.num_tokens, dtype=torch.bool)
        mask[torch.randperm(self.num_tokens, generator=generator)[:self.mask_count]] = True
        return fields, mask
