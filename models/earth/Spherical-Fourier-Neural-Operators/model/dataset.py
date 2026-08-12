"""Pair and split the deterministic spherical sequence for SFNO training."""

from torch.utils.data import Dataset


class SphericalPairDataset(Dataset):
    def __init__(self, fields, start, stop):
        self.fields = fields
        self.start = start
        self.stop = min(stop, fields.shape[0] - 1)
        if self.stop <= self.start:
            raise ValueError("A spherical split must contain at least one input/target pair")

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        step = self.start + index
        return self.fields[step], self.fields[step + 1]
