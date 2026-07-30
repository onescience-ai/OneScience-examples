#!/usr/bin/env python3
"""Validate the IBM/NASA Prithvi-EO Sen1Floods11 flood-segmentation model.

Despite the historical filename ``train.py``, this script does not retrain the
model. It reconstructs the exact published MMSegmentation architecture with
plain PyTorch, strictly loads the official checkpoint, runs inference, computes
all available metrics, and writes reproducible artifacts.

The plain-PyTorch implementation avoids the legacy runtime requirement
(mmcv-full 1.6.2 + mmsegmentation 0.30.0), which is not compatible with many
modern PyTorch/CUDA notebook environments.

Default usage from the model folder::

    python train.py --device cuda

For a real six-band GeoTIFF and optional label::

    python train.py --device cuda --input image.tif --label label.tif

Input band order must be Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2. If an
input has more than six bands, the official zero-based indexes 1,2,3,8,11,12
are selected by default. Use ``--bands 0,1,2,3,4,5`` to override this.

Required packages: torch, numpy
For GeoTIFF data/output: rasterio
Optional plotting: matplotlib
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


MODEL_NAME = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11"
WEIGHT_NAME = "sen1floods11_Prithvi_100M.pth"
EXPECTED_WEIGHT_SHA256 = (
    "c3b8af485ea03dab2a352c9269cb633a9f1550aa672733c1601306d4cd65b149"
)
TILE_SIZE = 224
STRIDE = 112
BAND_NAMES = ("Blue", "Green", "Red", "Narrow NIR", "SWIR 1", "SWIR 2")
OFFICIAL_FULL_IMAGE_BAND_INDEXES = (1, 2, 3, 8, 11, 12)  # zero-based
MEANS = np.asarray(
    [0.14245495, 0.13921481, 0.12434631, 0.31420089, 0.20743526, 0.12046503],
    dtype=np.float32,
)
STDS = np.asarray(
    [0.04036231, 0.04186983, 0.05267646, 0.08222210, 0.06834774, 0.05294205],
    dtype=np.float32,
)


class Attention(nn.Module):
    """Self-attention matching the timm block used by the official model."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class Mlp(nn.Module):
    """Transformer MLP with checkpoint-compatible parameter names."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """ViT block equivalent to the timm implementation used for training."""

    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Six-channel, one-timestep 3-D patch embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            6,
            768,
            kernel_size=(1, 16, 16),
            stride=(1, 16, 16),
            bias=True,
        )
        self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if tuple(x.shape[-2:]) != (TILE_SIZE, TILE_SIZE):
            raise ValueError(
                f"Each model tile must be 224x224, received {tuple(x.shape[-2:])}."
            )
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class TemporalViTEncoder(nn.Module):
    """Exact 12-layer Prithvi-100M encoder used by the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 197, 768), requires_grad=False
        )
        self.patch_embed = PatchEmbed()
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(768)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(
            x.shape[0], -1, -1
        )
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        return (self.norm(x),)


class Norm2d(nn.Module):
    """LayerNorm over channels, matching the official geospatial neck."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    """Four 2x transposed convolutions mapping 14x14 tokens to 224x224."""

    def __init__(self) -> None:
        super().__init__()
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2),
            Norm2d(768),
            nn.GELU(),
            nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2),
            Norm2d(768),
            nn.GELU(),
            nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2),
        )

    def forward(self, features: Tuple[Tensor]) -> Tuple[Tensor]:
        x = features[0][:, 1:, :]
        x = x.permute(0, 2, 1).reshape(x.shape[0], 768, 14, 14)
        return (self.fpn2(self.fpn1(x)),)


class ConvBNReLU(nn.Module):
    """MMSeg ConvModule subset with identical state-dict names."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class FCNHead(nn.Module):
    """MMSeg FCNHead used for the main and auxiliary segmentation heads."""

    def __init__(self, num_convs: int) -> None:
        super().__init__()
        layers: List[nn.Module] = [ConvBNReLU(768, 256)]
        layers.extend(ConvBNReLU(256, 256) for _ in range(num_convs - 1))
        self.convs = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, features: Tuple[Tensor]) -> Tensor:
        x = self.convs(features[-1])
        return self.conv_seg(self.dropout(x))


class PrithviFloodSegmenter(nn.Module):
    """Checkpoint-compatible end-to-end flood segmentation network."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = TemporalViTEncoder()
        self.neck = ConvTransformerTokensToEmbeddingNeck()
        self.decode_head = FCNHead(num_convs=1)
        # Present for strict checkpoint validation; not used during inference.
        self.auxiliary_head = FCNHead(num_convs=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode_head(self.neck(self.backbone(x)))


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run Prithvi Sen1Floods11 inference and validation."
    )
    parser.add_argument(
        "--model-dir", type=Path, default=script_dir, help="Model folder."
    )
    parser.add_argument(
        "--weights", type=Path, default=None, help=f"Path to {WEIGHT_NAME}."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input GeoTIFF; default searches the model folder, then uses synthetic data.",
    )
    parser.add_argument(
        "--label", type=str, default=None, help="Optional label GeoTIFF (0, 1, -1)."
    )
    parser.add_argument(
        "--bands",
        type=str,
        default=None,
        help="Comma-separated zero-based input band indexes, e.g. 0,1,2,3,4,5.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Result folder."
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument(
        "--skip-checksum", action="store_true", help="Skip SHA-256 integrity check."
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load_official(path: Path) -> Dict[str, Any]:
    """Load the trusted official IBM/NASA pickle, using mmap when supported."""

    kwargs: Dict[str, Any] = {
        "map_location": "cpu",
        # Required on PyTorch >=2.6 because this official legacy checkpoint also
        # stores NumPy metadata and optimizer state.
        "weights_only": False,
    }
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        try:
            return torch.load(path, **kwargs)
        except TypeError:  # PyTorch versions before weights_only was added.
            return torch.load(path, map_location="cpu")


def load_model(
    weights_path: Path, device: torch.device, check_checksum: bool
) -> Tuple[PrithviFloodSegmenter, Dict[str, Any]]:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    checksum = sha256_file(weights_path) if check_checksum else "skipped"
    if check_checksum and checksum != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch; the 1.2 GB download may be incomplete. "
            f"Expected {EXPECTED_WEIGHT_SHA256}, got {checksum}."
        )

    checkpoint = torch_load_official(weights_path)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise TypeError("Official checkpoint does not contain a state_dict.")
    state = checkpoint["state_dict"]
    checkpoint_meta = checkpoint.get("meta", {})

    model = PrithviFloodSegmenter()
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    shape_errors = [
        {
            "key": key,
            "checkpoint": list(state[key].shape),
            "model": list(model_state[key].shape),
        }
        for key in sorted(set(state) & set(model_state))
        if tuple(state[key].shape) != tuple(model_state[key].shape)
    ]
    if missing or unexpected or shape_errors:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={missing}, unexpected={unexpected}, shape_errors={shape_errors}"
        )
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    model.to(device).eval()

    metadata = {
        "checkpoint": str(weights_path.resolve()),
        "checkpoint_sha256": checksum,
        "checksum_matches_official": (
            None if checksum == "skipped" else checksum == EXPECTED_WEIGHT_SHA256
        ),
        "strict_load": True,
        "state_dict_key_count": len(model_state),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_epoch": checkpoint_meta.get("epoch"),
        "checkpoint_mmseg_version": checkpoint_meta.get("mmseg_version"),
        "checkpoint_mmcv_version": checkpoint_meta.get("mmcv_version"),
    }
    return model, metadata


def excluded_path(path: Path, output_dir: Path) -> bool:
    hidden_or_results = {".cache", ".hfdeps", ".official_source", output_dir.name}
    return any(part in hidden_or_results for part in path.parts)


def find_local_input(model_dir: Path, output_dir: Path) -> Optional[Path]:
    candidates = sorted(
        path
        for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
        for path in model_dir.rglob(pattern)
        if not excluded_path(path, output_dir)
        and "label" not in path.name.lower()
        and "pred" not in path.name.lower()
    )
    preferred = [p for p in candidates if "s2hand" in p.name.lower()]
    return (preferred or candidates or [None])[0]


def infer_label_path(input_path: Path, model_dir: Path) -> Optional[Path]:
    replacements = (
        input_path.with_name(input_path.name.replace("_S2Hand", "_LabelHand")),
        input_path.with_name(input_path.name.replace("S2Hand", "LabelHand")),
        model_dir
        / "LabelHand"
        / input_path.name.replace("_S2Hand", "_LabelHand"),
    )
    for path in replacements:
        if path != input_path and path.is_file():
            return path
    target_name = input_path.name.replace("_S2Hand", "_LabelHand").lower()
    for path in model_dir.rglob("*"):
        if path.is_file() and path.name.lower() == target_name:
            return path
    return None


def parse_band_indexes(spec: Optional[str], band_count: int) -> Tuple[int, ...]:
    if spec:
        indexes = tuple(int(value.strip()) for value in spec.split(","))
    elif band_count == 6:
        indexes = tuple(range(6))
    elif band_count > max(OFFICIAL_FULL_IMAGE_BAND_INDEXES):
        indexes = OFFICIAL_FULL_IMAGE_BAND_INDEXES
    else:
        raise ValueError(
            f"Input has {band_count} bands. Need six bands or enough bands for "
            f"indexes {OFFICIAL_FULL_IMAGE_BAND_INDEXES}."
        )
    if len(indexes) != 6 or len(set(indexes)) != 6:
        raise ValueError("Exactly six unique zero-based band indexes are required.")
    if min(indexes) < 0 or max(indexes) >= band_count:
        raise IndexError(f"Band indexes {indexes} are invalid for {band_count} bands.")
    return indexes


def load_geotiff(
    path: Path, bands_spec: Optional[str]
) -> Tuple[np.ndarray, Dict[str, Any], Tuple[int, ...]]:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(
            "Reading GeoTIFF requires rasterio. Install it with: pip install rasterio"
        ) from exc

    with rasterio.open(path) as src:
        indexes = parse_band_indexes(bands_spec, src.count)
        # rasterio uses one-based indexes.
        raw = src.read([index + 1 for index in indexes]).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        raw[np.isclose(raw, nodata)] = 0.0
    raw[raw == -9999] = 0.0
    return raw, profile, indexes


def load_label(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Reading a label GeoTIFF requires rasterio.") from exc
    with rasterio.open(path) as src:
        label = src.read(1)
        nodata = src.nodata
    label = label.astype(np.int64)
    if nodata is not None:
        label[label == int(nodata)] = -1
    return label


def make_synthetic(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create deterministic Sentinel-2-like data and a pseudo flood mask."""

    rng = np.random.default_rng(seed)
    height = width = TILE_SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    river_center = height * (0.52 + 0.08 * np.sin(xx / 24.0))
    river = np.abs(yy - river_center) < 14
    lake = (xx - 166) ** 2 + (yy - 67) ** 2 < 27**2
    label = np.where(river | lake, 1, 0).astype(np.int64)

    reflectance = MEANS[:, None, None] + rng.normal(
        0.0, STDS[:, None, None] * 0.45, size=(6, height, width)
    ).astype(np.float32)
    water_signature = np.asarray(
        [0.085, 0.095, 0.070, 0.040, 0.025, 0.015], dtype=np.float32
    )
    water_noise = rng.normal(0.0, 0.008, size=(6, height, width)).astype(np.float32)
    reflectance[:, label == 1] = (
        water_signature[:, None] + water_noise[:, label == 1]
    )
    reflectance = np.clip(reflectance, 0.0, 1.0)

    # A small no-data corner also verifies ignore-mask handling.
    label[:8, :8] = -1
    raw = np.rint(reflectance * 10000.0).astype(np.float32)
    raw[:, :8, :8] = -9999.0
    return raw, label


def preprocess(raw: np.ndarray) -> Tensor:
    if raw.ndim != 3 or raw.shape[0] != 6:
        raise ValueError(f"Expected raw input shape (6,H,W), got {raw.shape}.")
    raw = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    raw[raw == -9999] = 0.0
    reflectance = raw * 0.0001
    normalized = (reflectance - MEANS[:, None, None]) / STDS[:, None, None]
    return torch.from_numpy(np.ascontiguousarray(normalized)).float()


def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.inference_mode()
def predict_sliding(
    model: PrithviFloodSegmenter,
    normalized: Tensor,
    device: torch.device,
    stride: int,
) -> Tensor:
    """Official-style 224px sliding inference with 112px default overlap."""

    if not 1 <= stride <= TILE_SIZE:
        raise ValueError("--stride must be between 1 and 224.")
    _, original_h, original_w = normalized.shape
    padded_h = max(original_h, TILE_SIZE)
    padded_w = max(original_w, TILE_SIZE)
    padded = F.pad(normalized, (0, padded_w - original_w, 0, padded_h - original_h))
    ys = tile_starts(padded_h, TILE_SIZE, stride)
    xs = tile_starts(padded_w, TILE_SIZE, stride)
    logits_sum = torch.zeros((2, padded_h, padded_w), device=device)
    counts = torch.zeros((1, padded_h, padded_w), device=device)
    padded = padded.to(device)
    for y in ys:
        for x in xs:
            tile = padded[:, y : y + TILE_SIZE, x : x + TILE_SIZE]
            tile = tile.unsqueeze(0).unsqueeze(2)  # B,C,T,H,W
            logits = model(tile)[0]
            logits_sum[:, y : y + TILE_SIZE, x : x + TILE_SIZE] += logits
            counts[:, y : y + TILE_SIZE, x : x + TILE_SIZE] += 1
    logits_sum /= counts
    return logits_sum[:, :original_h, :original_w]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    model: PrithviFloodSegmenter,
    normalized: Tensor,
    device: torch.device,
    stride: int,
    warmup: int,
    runs: int,
) -> Tuple[Tensor, List[float], float]:
    for _ in range(max(warmup, 0)):
        _ = predict_sliding(model, normalized, device, stride)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings: List[float] = []
    logits: Optional[Tensor] = None
    for _ in range(max(runs, 1)):
        synchronize(device)
        started = time.perf_counter()
        logits = predict_sliding(model, normalized, device, stride)
        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000.0)
    assert logits is not None
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    return logits.detach().cpu(), timings, peak_mb


def confusion_metrics(prediction: np.ndarray, label: np.ndarray) -> Dict[str, Any]:
    if prediction.shape != label.shape:
        raise ValueError(
            f"Prediction/label shape mismatch: {prediction.shape} vs {label.shape}."
        )
    valid = np.isin(label, (0, 1))
    total = int(valid.sum())
    if total == 0:
        raise ValueError("Label has no valid class-0 or class-1 pixels.")
    matrix = np.zeros((2, 2), dtype=np.int64)
    for truth in range(2):
        for pred in range(2):
            matrix[truth, pred] = int(((label == truth) & (prediction == pred)).sum())
    true_positive = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - true_positive
    truth_count = matrix.sum(axis=1)
    iou = np.divide(
        true_positive, union, out=np.full(2, np.nan), where=union > 0
    )
    accuracy = np.divide(
        true_positive,
        truth_count,
        out=np.full(2, np.nan),
        where=truth_count > 0,
    )
    return {
        "valid_pixels": total,
        "ignored_pixels": int(label.size - total),
        "confusion_matrix_rows_truth_cols_prediction": matrix.tolist(),
        "class_iou": {"no_water": float(iou[0]), "water_flood": float(iou[1])},
        "class_accuracy": {
            "no_water": float(accuracy[0]),
            "water_flood": float(accuracy[1]),
        },
        "overall_accuracy": float(true_positive.sum() / total),
        "mean_iou": float(np.nanmean(iou)),
        "mean_accuracy": float(np.nanmean(accuracy)),
    }


def save_prediction_geotiff(
    path: Path, prediction: np.ndarray, profile: Optional[Dict[str, Any]]
) -> Optional[str]:
    if profile is None:
        return None
    try:
        import rasterio
    except ImportError:
        return None
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="int16", nodata=-1, compress="lzw")
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(prediction.astype(np.int16), 1)
    return str(path.resolve())


def save_plot(
    path: Path,
    raw: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
    label: Optional[np.ndarray],
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    rgb = np.moveaxis(raw[[2, 1, 0]], 0, -1) * 0.0001
    lo, hi = np.nanpercentile(rgb[np.isfinite(rgb)], [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    columns = 4 if label is not None else 3
    figure, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.1))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[1].imshow(probabilities[1], cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Water probability")
    axes[2].imshow(prediction, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("Prediction (0/1)")
    if label is not None:
        axes[3].imshow(label, cmap="Blues", vmin=0, vmax=1)
        axes[3].set_title("Label / synthetic target")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(path.resolve())


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_summary(path: Path, results: Dict[str, Any]) -> None:
    metrics = results.get("accuracy_metrics") or {}
    lines = [
        "Prithvi-EO Sen1Floods11 validation",
        "=" * 42,
        f"Status: {results['status']}",
        f"Validation level: {results['validation_level']}",
        f"Data source: {results['data_source']}",
        f"Device: {results['environment']['device']}",
        f"Input shape: {tuple(results['input']['shape'])}",
        f"Output shape: {tuple(results['output']['shape'])}",
        f"Strict checkpoint load: {results['model']['strict_load']}",
        f"Finite output: {results['output']['finite_fraction']:.6f}",
        f"Median latency: {results['performance']['median_latency_ms']:.3f} ms",
        f"Peak GPU memory: {results['performance']['peak_gpu_memory_mb']:.2f} MB",
        f"Predicted water fraction: {results['output']['water_fraction']:.6f}",
        f"Mean confidence: {results['output']['mean_confidence']:.6f}",
    ]
    if metrics:
        qualifier = (
            "synthetic pseudo-label; not scientific model accuracy"
            if results["validation_level"] == "functional_only"
            else "paired ground-truth label"
        )
        lines.extend(
            [
                f"Metric basis: {qualifier}",
                f"Overall accuracy: {metrics['overall_accuracy']:.6f}",
                f"Mean IoU: {metrics['mean_iou']:.6f}",
                f"Mean accuracy: {metrics['mean_accuracy']:.6f}",
                f"No-water IoU: {metrics['class_iou']['no_water']:.6f}",
                f"Water/flood IoU: {metrics['class_iou']['water_flood']:.6f}",
            ]
        )
    else:
        lines.append("Scientific accuracy: unavailable (no paired label)")
    lines.append(f"Results: {results['output_dir']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = args.model_dir.resolve()
    weights_path = (
        args.weights.resolve() if args.weights else model_dir / WEIGHT_NAME
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else model_dir / "validation_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    print(f"Python:  {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {device}")
    print("Loading and strictly validating the official checkpoint...")
    model, model_metadata = load_model(
        weights_path, device, check_checksum=not args.skip_checksum
    )

    input_path: Optional[Path]
    if args.input and args.input.lower() != "none":
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input not found: {input_path}")
    elif args.input and args.input.lower() == "none":
        input_path = None
    else:
        input_path = find_local_input(model_dir, output_dir)

    profile: Optional[Dict[str, Any]] = None
    band_indexes: Tuple[int, ...] = tuple(range(6))
    label_path: Optional[Path] = None
    if input_path is None:
        raw, label = make_synthetic(args.seed)
        data_source = "synthetic_sentinel2_like"
        validation_level = "functional_only"
        np.save(output_dir / "synthetic_input.npy", raw)
        np.save(output_dir / "synthetic_pseudo_label.npy", label)
    else:
        raw, profile, band_indexes = load_geotiff(input_path, args.bands)
        if args.label:
            label_path = Path(args.label).expanduser().resolve()
            if not label_path.is_file():
                raise FileNotFoundError(f"Label not found: {label_path}")
        else:
            label_path = infer_label_path(input_path, model_dir)
        label = load_label(label_path) if label_path else None
        data_source = "local_geotiff"
        validation_level = "paired_scientific" if label is not None else "functional_only"

    normalized = preprocess(raw)
    logits, timings, peak_gpu_mb = benchmark(
        model,
        normalized,
        device,
        stride=args.stride,
        warmup=args.warmup,
        runs=args.runs,
    )
    probabilities_tensor = logits.softmax(dim=0)
    probabilities = probabilities_tensor.numpy()
    prediction = probabilities.argmax(axis=0).astype(np.int16)
    finite_fraction = float(torch.isfinite(logits).float().mean().item())
    accuracy_metrics = (
        confusion_metrics(prediction, label) if label is not None else None
    )

    np.save(output_dir / "logits.npy", logits.numpy())
    np.save(output_dir / "probabilities.npy", probabilities)
    np.save(output_dir / "prediction.npy", prediction)
    geotiff_output = save_prediction_geotiff(
        output_dir / "prediction.tif", prediction, profile
    )
    plot_output = None
    if not args.no_plot:
        plot_output = save_plot(
            output_dir / "validation_preview.png",
            raw,
            prediction,
            probabilities,
            label,
        )

    confidence = probabilities.max(axis=0)
    results: Dict[str, Any] = {
        "status": "PASS" if finite_fraction == 1.0 else "FAIL",
        "validation_level": validation_level,
        "model_name": MODEL_NAME,
        "data_source": data_source,
        "model": model_metadata,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "input": {
            "path": str(input_path) if input_path else None,
            "label_path": str(label_path) if label_path else None,
            "shape": list(raw.shape),
            "band_names": list(BAND_NAMES),
            "zero_based_source_band_indexes": list(band_indexes),
            "scale_factor": 0.0001,
            "normalization_means": MEANS.tolist(),
            "normalization_stds": STDS.tolist(),
            "tile_size": TILE_SIZE,
            "stride": args.stride,
        },
        "output": {
            "shape": list(logits.shape),
            "finite_fraction": finite_fraction,
            "water_fraction": float((prediction == 1).mean()),
            "mean_confidence": float(confidence.mean()),
            "minimum_confidence": float(confidence.min()),
            "maximum_confidence": float(confidence.max()),
            "prediction_npy": str((output_dir / "prediction.npy").resolve()),
            "prediction_geotiff": geotiff_output,
            "preview": plot_output,
        },
        "performance": {
            "warmup_runs": max(args.warmup, 0),
            "timed_runs": max(args.runs, 1),
            "latencies_ms": timings,
            "median_latency_ms": float(statistics.median(timings)),
            "mean_latency_ms": float(statistics.mean(timings)),
            "peak_gpu_memory_mb": float(peak_gpu_mb),
        },
        "accuracy_metrics": accuracy_metrics,
        "metric_interpretation": (
            "Metrics use a deterministic synthetic pseudo-label and only test the "
            "pipeline; they are not scientific accuracy."
            if data_source.startswith("synthetic")
            else (
                "Metrics use the paired local ground-truth label."
                if accuracy_metrics is not None
                else "No paired label was found; scientific accuracy is unavailable."
            )
        ),
        "official_reference_metrics": {
            "test_overall_accuracy": 0.9725,
            "test_mean_iou": 0.8868,
            "test_mean_accuracy": 0.9437,
            "test_no_water_iou": 0.9690,
            "test_water_flood_iou": 0.8046,
            "note": "Published reference only; not recomputed without Sen1Floods11 test data.",
        },
        "output_dir": str(output_dir.resolve()),
    }
    results = json_ready(results)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(output_dir / "summary.txt", results)

    print("\n========== Prithvi Sen1Floods11 validation ==========")
    print(f"Status:              {results['status']}")
    print(f"Validation level:    {validation_level}")
    print(f"Data source:         {data_source}")
    print(f"Device:              {device}")
    print(f"Input shape:         {tuple(raw.shape)}")
    print(f"Output shape:        {tuple(logits.shape)}")
    print(f"Strict load:         {model_metadata['strict_load']}")
    print(f"Parameters:          {model_metadata['parameter_count']:,}")
    print(f"Finite output:       {finite_fraction:.6f}")
    print(f"Median latency:      {statistics.median(timings):.3f} ms")
    print(f"Peak GPU memory:     {peak_gpu_mb:.2f} MB")
    print(f"Predicted water:     {(prediction == 1).mean():.6f}")
    print(f"Mean confidence:     {confidence.mean():.6f}")
    if accuracy_metrics is not None:
        prefix = "Synthetic" if data_source.startswith("synthetic") else "Paired-label"
        print(f"{prefix} mIoU:     {accuracy_metrics['mean_iou']:.6f}")
        print(f"{prefix} accuracy: {accuracy_metrics['overall_accuracy']:.6f}")
        if data_source.startswith("synthetic"):
            print("Scientific accuracy: unavailable (synthetic pseudo-label only)")
    else:
        print("Scientific accuracy: unavailable (no paired label)")
    print(f"Results:             {output_dir.resolve()}")
    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
