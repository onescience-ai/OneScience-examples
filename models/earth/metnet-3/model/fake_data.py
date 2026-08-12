"""Indexed compact MetNet-3 fake samples."""

import torch
from torch.utils.data import Dataset


def make_fake(config, batch_size=2, height=8, width=8, seed=7):
    torch.manual_seed(seed)
    def field(t, c, h=height, w=width): return torch.randn(batch_size, t, c, h, w)
    batch = {"mrms_high": field(config.high_frames, config.high_channels), "mrms_low": field(2, config.low_channels, height // 2, width // 2), "omo": field(config.omo_frames, config.omo_channels), "hrrr": field(1, config.hrrr_channels), "goes": field(1, config.goes_channels), "elevation": field(1, 1, height, width)[:, 0], "coordinates": field(1, 2, height, width)[:, 0], "topography_embedding": field(1, 1, height, width)[:, 0], "current_time": torch.rand(batch_size, 1), "lead_time": torch.rand(batch_size, 1), "omo_input_mask": torch.rand(batch_size, 1, height, width) > .25}
    targets = {"precipitation": torch.randint(config.precipitation_bins, (batch_size, 1, height, width)), "ground": torch.randint(config.ground_bins, (batch_size, config.ground_targets, height, width)), "hrrr": torch.randn(batch_size, config.hrrr_channels, height, width), "precipitation_mask": torch.ones(batch_size, 1, height, width), "ground_mask": torch.ones(batch_size, config.ground_targets, height, width), "hrrr_mask": torch.ones(batch_size, config.hrrr_channels, height, width)}
    return batch, targets


class FakeMetNetDataset(Dataset):
    def __init__(self, config, samples, seed): self.config, self.samples, self.seed = config, samples, seed
    def __len__(self): return self.samples
    def __getitem__(self, index):
        batch, targets = make_fake(self.config, batch_size=1, seed=self.seed + index)
        return {key: value.squeeze(0) for key, value in batch.items()}, {key: value.squeeze(0) for key, value in targets.items()}
