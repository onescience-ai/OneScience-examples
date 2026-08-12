"""Indexed deterministic cubed-sphere samples for training and validation."""

from torch.utils.data import Dataset
from .fake_data import make_fake_batch


class FakeCubeSphereDataset(Dataset):
    def __init__(self, samples=64, seed=7, channels=2, height=8, width=8):
        self.samples, self.seed = samples, seed
        self.shape = (channels, 6, height, width)

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return make_fake_batch(batch=1, channels=self.shape[0], faces=6, height=self.shape[2], width=self.shape[3], seed=self.seed + index)[0]
