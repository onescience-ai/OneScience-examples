import os
import sys
from pathlib import Path

import dgl
import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from onescience.utils.YParams import YParams


def make_graph(num_nodes: int = 12):
    src = torch.arange(num_nodes, dtype=torch.int32)
    dst = torch.roll(src, shifts=-1)
    graph = dgl.to_bidirected(dgl.graph((src, dst), num_nodes=num_nodes, idtype=torch.int32))

    pos = torch.stack(
        (
            torch.linspace(0.0, 1.0, num_nodes),
            torch.sin(torch.linspace(0.0, 3.14159, num_nodes)) * 0.2,
        ),
        dim=1,
    )
    row, col = graph.edges()
    disp = pos[row.long()] - pos[col.long()]
    graph.edata["x"] = torch.cat(
        (disp, torch.linalg.norm(disp, dim=-1, keepdim=True)),
        dim=1,
    )

    velocity = torch.randn(num_nodes, 2) * 0.1
    node_type = torch.zeros(num_nodes, 4)
    node_type[:, 0] = 1.0
    graph.ndata["x"] = torch.cat((velocity, node_type), dim=1)
    graph.ndata["y"] = torch.cat(
        (torch.randn(num_nodes, 2) * 0.01, torch.randn(num_nodes, 1) * 0.01),
        dim=1,
    )
    graph.ndata["mesh_pos"] = pos

    cells = torch.tensor(
        [[i, i + 1, min(i + 2, num_nodes - 1)] for i in range(num_nodes - 2)],
        dtype=torch.int64,
    )
    mask = torch.ones(num_nodes, 1, dtype=torch.bool)
    return {"graph": graph, "cells": cells, "mask": mask}


class FakeGraphDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if isinstance(sample, dict) and "graph" in sample:
            return sample["graph"]
        return sample


def _resolve_path(project_root: Path, path):
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class FakeCylinderFlowDatapipe:
    def __init__(self, params, project_root: Path):
        self.params = params
        fake_data_path = _resolve_path(project_root, params.source.fake_data_path)
        if not fake_data_path.exists():
            raise FileNotFoundError(
                f"Fake data file not found: {fake_data_path}. Run scripts/fake_data.py first."
            )

        payload = _torch_load(fake_data_path)
        self.train_dataset = FakeGraphDataset(payload["train"])
        self.val_dataset = FakeGraphDataset(payload["val"])
        self.test_dataset = FakeGraphDataset(payload["test"])
        self.stats = payload.get("stats", {})

    def _loader(self, dataset, shuffle=False, drop_last=False):
        return GraphDataLoader(
            dataset,
            batch_size=self.params.dataloader.batch_size,
            drop_last=drop_last,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True), None

    def val_dataloader(self):
        return self._loader(self.val_dataset), None

    def test_dataloader(self):
        return self._loader(self.test_dataset)


def use_fake_data(params):
    return bool(getattr(params.source, "fake_data", False))


def build_cylinder_flow_datapipe(params, distributed: bool, project_root: Path):
    if use_fake_data(params):
        return FakeCylinderFlowDatapipe(params=params, project_root=project_root)

    from onescience.datapipes.cfd import DeepMind_CylinderFlowDatapipe

    return DeepMind_CylinderFlowDatapipe(params=params, distributed=distributed)


def main():
    os.chdir(PROJECT_ROOT)
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    cfg_data = YParams(config_path, "datapipe")
    output_path = PROJECT_ROOT / cfg_data.source.fake_data_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "train": [
            make_graph()
            for _ in range(cfg_data.data.train_samples * (cfg_data.data.train_steps - 1))
        ],
        "val": [
            make_graph()
            for _ in range(cfg_data.data.val_samples * (cfg_data.data.val_steps - 1))
        ],
        "test": [
            make_graph()
            for _ in range(cfg_data.data.test_samples * (cfg_data.data.test_steps - 1))
        ],
        "stats": {
            "edge_stats": {
                "edge_mean": torch.zeros(3),
                "edge_std": torch.ones(3),
            },
            "node_stats": {
                "velocity_mean": torch.zeros(2),
                "velocity_std": torch.ones(2),
                "velocity_diff_mean": torch.zeros(2),
                "velocity_diff_std": torch.ones(2),
                "pressure_mean": torch.zeros(1),
                "pressure_std": torch.ones(1),
            },
        },
    }
    torch.save(payload, output_path)
    print(f"Fake data saved to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
