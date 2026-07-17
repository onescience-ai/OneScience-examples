from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.fno import FNO1d, FNO2d, FNO3d, FNO_maxwell

try:
    from onescience.utils.YParams import YParams
except ModuleNotFoundError as exc:
    YParams = None
    YPARAMS_IMPORT_ERROR = exc

try:
    from onescience.distributed.manager import DistributedManager
except ModuleNotFoundError as exc:
    DistributedManager = None
    DISTRIBUTED_IMPORT_ERROR = exc

try:
    from onescience.datapipes.cfd.PDENNEval import PDEBenchFNODatapipe
except (ImportError, ModuleNotFoundError, OSError) as exc:
    PDEBenchFNODatapipe = None
    DATAPIPE_IMPORT_ERROR = exc


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def set_attr(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def load_config(config_path: Path | str = DEFAULT_CONFIG) -> Any:
    config_path = Path(config_path).expanduser().resolve()
    if YParams is not None:
        return YParams(str(config_path), "fno_config")

    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or "fno_config" not in raw:
        raise ValueError(f"{config_path} must contain root key 'fno_config'")
    return to_namespace(raw["fno_config"])


def resolve_path(value: Any, root: Path = PROJECT_ROOT) -> Path:
    if value is None:
        raise ValueError("path value can not be null")
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def prepare_config(
    cfg: Any,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> Any:
    source = cfg.datapipe.source
    training = cfg.training
    inference = get_attr(cfg, "inference", None)

    resolved_data = resolve_path(data_dir or source.data_dir)
    set_attr(source, "data_dir", str(resolved_data))

    resolved_output = resolve_path(output_dir or training.output_dir)
    set_attr(training, "output_dir", str(resolved_output))

    model_path = get_attr(training, "model_path", None)
    if model_path:
        set_attr(training, "model_path", str(resolve_path(model_path)))

    if inference is not None:
        infer_output = get_attr(inference, "output_dir", "./result/output")
        infer_checkpoint = checkpoint or get_attr(inference, "checkpoint", None)
        set_attr(inference, "output_dir", str(resolve_path(infer_output)))
        if infer_checkpoint:
            set_attr(inference, "checkpoint", str(resolve_path(infer_checkpoint)))

    return cfg


class SingleProcessManager:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_distributed() -> Any:
    if DistributedManager is None:
        warnings.warn(
            f"Falling back to a single-process manager: {DISTRIBUTED_IMPORT_ERROR}",
            RuntimeWarning,
        )
        return SingleProcessManager()
    DistributedManager.initialize()
    return DistributedManager()


def cleanup_distributed() -> None:
    if DistributedManager is not None and DistributedManager.is_initialized():
        DistributedManager.cleanup()


class LocalPDEBenchFNODataset(Dataset):
    """Small FNO-compatible reader used when optional OneScience graph deps are absent."""

    def __init__(self, datapipe_cfg: Any, mode: str):
        self.mode = mode
        self.data_cfg = datapipe_cfg.data
        self.source_cfg = datapipe_cfg.source
        self.initial_step = int(self.data_cfg.initial_step)
        self.reduced_resolution = int(self.data_cfg.reduced_resolution)
        self.reduced_resolution_t = int(self.data_cfg.reduced_resolution_t)
        self.reduced_batch = int(self.data_cfg.reduced_batch)
        self.test_ratio = float(get_attr(self.data_cfg, "test_ratio", 0.1))
        self.file_path = Path(self.source_cfg.data_dir) / self.source_cfg.file_name
        self._load_single_file()

    def _load_single_file(self) -> None:
        if not self.file_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")

        with h5py.File(self.file_path, "r") as handle:
            if "tensor" not in handle:
                raise ValueError("local fallback datapipe requires HDF5 dataset 'tensor'")
            tensor = np.asarray(handle["tensor"], dtype=np.float32)
            if tensor.ndim == 3:
                data = tensor[
                    :: self.reduced_batch,
                    :: self.reduced_resolution_t,
                    :: self.reduced_resolution,
                ]
                data = np.transpose(data, (0, 2, 1))
                self.data = data[:, :, :, None]
                x = np.asarray(handle["x-coordinate"], dtype=np.float32)
                self.grid = torch.tensor(
                    x[:: self.reduced_resolution], dtype=torch.float32
                ).unsqueeze(-1)
            elif tensor.ndim == 4:
                data = tensor[
                    :: self.reduced_batch,
                    :,
                    :: self.reduced_resolution,
                    :: self.reduced_resolution,
                ]
                data = np.transpose(data, (0, 2, 3, 1))
                if "nu" in handle:
                    nu = np.asarray(handle["nu"], dtype=np.float32)[
                        :: self.reduced_batch,
                        :: self.reduced_resolution,
                        :: self.reduced_resolution,
                    ]
                    data = np.concatenate([nu[:, :, :, None], data], axis=-1)
                self.data = data[:, :, :, :, None]
                x = torch.tensor(np.asarray(handle["x-coordinate"], dtype=np.float32))
                y = torch.tensor(np.asarray(handle["y-coordinate"], dtype=np.float32))
                xx, yy = torch.meshgrid(x, y, indexing="ij")
                self.grid = torch.stack((xx, yy), dim=-1)[
                    :: self.reduced_resolution, :: self.reduced_resolution
                ].float()
            else:
                raise ValueError(f"unsupported tensor ndim for FNO fallback: {tensor.ndim}")

        sample_count = self.data.shape[0]
        val_count = max(1, int(sample_count * self.test_ratio)) if sample_count > 1 else 0
        if self.mode == "train":
            self.data = self.data[val_count:]
        else:
            self.data = self.data[:val_count]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.spatial_dim = len(self.data.shape) - 3

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        return sample[..., : self.initial_step, :], sample, self.grid


class LocalPDEBenchFNODatapipe:
    def __init__(self, cfg: Any, distributed: bool = False):
        self.config = cfg
        self.distributed = distributed
        self.train_dataset = LocalPDEBenchFNODataset(cfg.datapipe, "train")
        self.val_dataset = LocalPDEBenchFNODataset(cfg.datapipe, "val")
        self.spatial_dim = self.train_dataset.spatial_dim

    def train_dataloader(self):
        loader_args = self.config.datapipe.dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=int(loader_args.batch_size),
            num_workers=int(loader_args.num_workers),
            pin_memory=bool(loader_args.pin_memory),
            shuffle=True,
            drop_last=True,
        ), None

    def val_dataloader(self):
        loader_args = self.config.datapipe.dataloader
        return DataLoader(
            self.val_dataset,
            batch_size=int(loader_args.batch_size),
            num_workers=int(loader_args.num_workers),
            pin_memory=bool(loader_args.pin_memory),
            shuffle=False,
            drop_last=False,
        ), None


def build_datapipe(cfg: Any, distributed: bool = False, force_local: bool = False) -> Any:
    if PDEBenchFNODatapipe is not None and not force_local:
        return PDEBenchFNODatapipe(cfg, distributed=distributed)
    if PDEBenchFNODatapipe is None:
        warnings.warn(
            f"Using local FNO HDF5 datapipe because OneScience datapipe import failed: "
            f"{DATAPIPE_IMPORT_ERROR}",
            RuntimeWarning,
        )
    return LocalPDEBenchFNODatapipe(cfg, distributed=distributed)


def build_model(spatial_dim: int, cfg: Any) -> torch.nn.Module:
    model_args = cfg.model
    data_cfg = cfg.datapipe.data
    initial_step = int(data_cfg.initial_step)
    pde_name = get_attr(data_cfg, "pde_name", "")
    modes = int(model_args.modes)

    if pde_name == "3D_Maxwell":
        return FNO_maxwell(
            num_channels=int(model_args.num_channels),
            width=int(model_args.width),
            modes1=modes,
            modes2=modes,
            modes3=modes,
            initial_step=initial_step,
        )
    if spatial_dim == 1:
        return FNO1d(
            num_channels=int(model_args.num_channels),
            width=int(model_args.width),
            modes=modes,
            initial_step=initial_step,
        )
    if spatial_dim == 2:
        return FNO2d(
            num_channels=int(model_args.num_channels),
            width=int(model_args.width),
            modes1=modes,
            modes2=modes,
            initial_step=initial_step,
        )
    if spatial_dim == 3:
        return FNO3d(
            num_channels=int(model_args.num_channels),
            width=int(model_args.width),
            modes1=modes,
            modes2=modes,
            modes3=modes,
            initial_step=initial_step,
        )
    raise ValueError(f"unsupported spatial dimension: {spatial_dim}")


def predict_batch(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, grid: torch.Tensor, cfg: Any):
    data_cfg = cfg.datapipe.data
    train_cfg = cfg.training
    initial_step = int(data_cfg.initial_step)
    t_train = min(int(train_cfg.t_train), y.shape[-2])
    input_shape = list(x.shape)[:-2] + [-1]

    if get_attr(train_cfg, "training_type", "single") == "autoregressive":
        pred = y[..., :initial_step, :]
        for _ in range(initial_step, t_train):
            model_input = x.reshape(input_shape)
            model_output = model(model_input, grid)
            if model_output.dim() == pred.dim() - 1:
                model_output = model_output.unsqueeze(-2)
            pred = torch.cat((pred, model_output), dim=-2)
            x = torch.cat((x[..., 1:, :], model_output), dim=-2)
        return pred, y[..., :t_train, :]

    model_input = x.reshape(input_shape)
    target = y[..., t_train - 1 : t_train, :]
    pred = model(model_input, grid)
    if pred.dim() == target.dim() - 1:
        pred = pred.unsqueeze(-2)
    return pred, target


def load_model_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"unsupported checkpoint format: {path}")
