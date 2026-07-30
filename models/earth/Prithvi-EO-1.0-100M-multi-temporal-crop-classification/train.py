#!/usr/bin/env python3
"""Validate the IBM/NASA Prithvi multi-temporal crop classifier.

Despite the historical filename ``train.py``, this script does not retrain the
model. It reconstructs the complete published MMSegmentation architecture with
plain PyTorch, strictly loads the official checkpoint, runs inference, computes
all available metrics, and writes reproducible artifacts.

Default usage::

    python train.py --device cuda

Real-data usage::

    python train.py --device cuda --input chip_001_merged.tif \
        --label chip_001.mask.tif

The official input is an 18-band, 224x224 HLS GeoTIFF containing six spectral
bands over three observations. This implementation deliberately follows the
published preprocessing configuration exactly: normalize the flat 18-channel
tensor, then reshape it to (6 bands, 3 time steps, H, W).

Required packages: torch, numpy
For GeoTIFF data/output: rasterio
Optional plotting: matplotlib
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


MODEL_NAME = (
    "ibm-nasa-geospatial/"
    "Prithvi-EO-1.0-100M-multi-temporal-crop-classification"
)
WEIGHT_NAME = "multi_temporal_crop_classification_Prithvi_100M.pth"
EXPECTED_WEIGHT_SHA256 = (
    "37ed41637eccccec65ca2031324e2c03a4f168e1ea0ea71ad180910589fa018c"
)
TILE_SIZE = 224
STRIDE = 112
NUM_FRAMES = 3
NUM_BANDS = 6
NUM_INPUT_CHANNELS = NUM_FRAMES * NUM_BANDS
NUM_CLASSES = 13
BAND_NAMES = ("Blue", "Green", "Red", "Narrow NIR", "SWIR 1", "SWIR 2")
CLASS_NAMES = (
    "Natural Vegetation",
    "Forest",
    "Corn",
    "Soybeans",
    "Wetlands",
    "Developed/Barren",
    "Open Water",
    "Winter Wheat",
    "Alfalfa",
    "Fallow/Idle Cropland",
    "Cotton",
    "Sorghum",
    "Other",
)
BASE_MEANS = np.asarray(
    [494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962, 1739.579917],
    dtype=np.float32,
)
BASE_STDS = np.asarray(
    [284.925432, 357.848760, 575.566823, 896.601013, 951.900334, 921.407808],
    dtype=np.float32,
)
# Exact sequence from the official MMSeg configuration.
MEANS_18 = np.tile(BASE_MEANS, NUM_FRAMES)
STDS_18 = np.tile(BASE_STDS, NUM_FRAMES)


class Attention(nn.Module):
    """Self-attention matching the timm block used by the official model."""

    def __init__(self, dim: int = 768, num_heads: int = 8) -> None:
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
    def __init__(self, dim: int = 768, hidden_dim: int = 3072) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(768)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(768)
        self.mlp = Mlp()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    """Six-channel, three-timestep 3-D patch embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            NUM_BANDS,
            768,
            kernel_size=(1, 16, 16),
            stride=(1, 16, 16),
            bias=True,
        )
        self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or tuple(x.shape[1:3]) != (NUM_BANDS, NUM_FRAMES):
            raise ValueError(
                "Expected tile shape (B,6,3,224,224), got " f"{tuple(x.shape)}."
            )
        if tuple(x.shape[-2:]) != (TILE_SIZE, TILE_SIZE):
            raise ValueError(
                f"Each model tile must be 224x224, got {tuple(x.shape[-2:])}."
            )
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class TemporalViTEncoder(nn.Module):
    """Six-layer temporal Prithvi encoder used by this checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, NUM_FRAMES * 14 * 14 + 1, 768), requires_grad=False
        )
        self.patch_embed = PatchEmbed()
        self.blocks = nn.ModuleList([Block() for _ in range(6)])
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
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    """Reshape three temporal token grids and upscale 14x14 to 224x224."""

    def __init__(self) -> None:
        super().__init__()
        dim = 768 * NUM_FRAMES
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            Norm2d(dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            Norm2d(dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )

    def forward(self, features: Tuple[Tensor]) -> Tuple[Tensor]:
        x = features[0][:, 1:, :]
        x = x.permute(0, 2, 1).reshape(x.shape[0], 768 * NUM_FRAMES, 14, 14)
        return (self.fpn2(self.fpn1(x)),)


class ConvBNReLU(nn.Module):
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
    def __init__(self, num_convs: int) -> None:
        super().__init__()
        layers: List[nn.Module] = [ConvBNReLU(768 * NUM_FRAMES, 256)]
        layers.extend(ConvBNReLU(256, 256) for _ in range(num_convs - 1))
        self.convs = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    def forward(self, features: Tuple[Tensor]) -> Tensor:
        return self.conv_seg(self.dropout(self.convs(features[-1])))


class PrithviCropSegmenter(nn.Module):
    """Checkpoint-compatible end-to-end 13-class crop segmenter."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = TemporalViTEncoder()
        self.neck = ConvTransformerTokensToEmbeddingNeck()
        self.decode_head = FCNHead(num_convs=1)
        # Required for strict checkpoint validation; not used for inference.
        self.auxiliary_head = FCNHead(num_convs=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode_head(self.neck(self.backbone(x)))


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run Prithvi multi-temporal crop-classification validation."
    )
    parser.add_argument("--model-dir", type=Path, default=script_dir)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="18-band input GeoTIFF; default searches locally then uses synthetic data.",
    )
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--bands",
        type=str,
        default=None,
        help="Optional 18 comma-separated zero-based source-band indexes.",
    )
    parser.add_argument(
        "--label-zero-based",
        action="store_true",
        help="Treat real labels as 0..12 with negative ignore; default is official 0=no-data, 1..13 classes.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load_official(path: Path) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        try:
            return torch.load(path, **kwargs)
        except TypeError:
            return torch.load(path, map_location="cpu")


def load_model(
    weights_path: Path, device: torch.device, check_checksum: bool
) -> Tuple[PrithviCropSegmenter, Dict[str, Any]]:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    checksum = sha256_file(weights_path) if check_checksum else "skipped"
    if check_checksum and checksum != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch; download may be incomplete. "
            f"Expected {EXPECTED_WEIGHT_SHA256}, got {checksum}."
        )
    checkpoint = torch_load_official(weights_path)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise TypeError("Official checkpoint does not contain a state_dict.")
    state = checkpoint["state_dict"]
    checkpoint_meta = checkpoint.get("meta", {})

    model = PrithviCropSegmenter()
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_errors = [
        {
            "key": key,
            "checkpoint": list(state[key].shape),
            "model": list(expected[key].shape),
        }
        for key in sorted(set(state) & set(expected))
        if tuple(state[key].shape) != tuple(expected[key].shape)
    ]
    if missing or unexpected or shape_errors:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={missing}, unexpected={unexpected}, shape_errors={shape_errors}"
        )
    model.load_state_dict(state, strict=True)
    state_key_count = len(state)
    del state, checkpoint
    model.to(device).eval()
    metadata = {
        "checkpoint": str(weights_path.resolve()),
        "checkpoint_sha256": checksum,
        "checksum_matches_official": (
            None if checksum == "skipped" else checksum == EXPECTED_WEIGHT_SHA256
        ),
        "strict_load": True,
        "state_dict_key_count": state_key_count,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_epoch": checkpoint_meta.get("epoch"),
        "checkpoint_mmseg_version": checkpoint_meta.get("mmseg_version"),
        "checkpoint_mmcv_version": checkpoint_meta.get("mmcv_version"),
    }
    return model, metadata


def excluded_path(path: Path, output_dir: Path) -> bool:
    excluded = {".cache", ".hfdeps", "validation_results", output_dir.name}
    return any(part in excluded for part in path.parts)


def find_local_input(model_dir: Path, output_dir: Path) -> Optional[Path]:
    candidates = sorted(
        path
        for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
        for path in model_dir.rglob(pattern)
        if not excluded_path(path, output_dir)
        and ".mask." not in path.name.lower()
        and "pred" not in path.name.lower()
    )
    preferred = [p for p in candidates if "_merged" in p.name.lower()]
    return (preferred or candidates or [None])[0]


def infer_label_path(input_path: Path, model_dir: Path) -> Optional[Path]:
    names = [
        input_path.name.replace("_merged.tif", ".mask.tif"),
        input_path.name.replace("_merged.tiff", ".mask.tif"),
        input_path.stem + ".mask.tif",
    ]
    for name in names:
        candidate = input_path.with_name(name)
        if candidate.is_file() and candidate != input_path:
            return candidate
    lower_names = {name.lower() for name in names}
    for path in model_dir.rglob("*"):
        if path.is_file() and path.name.lower() in lower_names:
            return path
    return None


def parse_band_indexes(spec: Optional[str], band_count: int) -> Tuple[int, ...]:
    if spec:
        indexes = tuple(int(value.strip()) for value in spec.split(","))
    elif band_count == NUM_INPUT_CHANNELS:
        indexes = tuple(range(NUM_INPUT_CHANNELS))
    elif band_count > NUM_INPUT_CHANNELS:
        indexes = tuple(range(NUM_INPUT_CHANNELS))
    else:
        raise ValueError(f"Input has {band_count} bands; at least 18 are required.")
    if len(indexes) != NUM_INPUT_CHANNELS or len(set(indexes)) != len(indexes):
        raise ValueError("Exactly 18 unique zero-based band indexes are required.")
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
        raw = src.read([index + 1 for index in indexes]).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        raw[np.isclose(raw, nodata)] = 0.0
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw, profile, indexes


def load_label(path: Path, zero_based: bool) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Reading a label GeoTIFF requires rasterio.") from exc
    with rasterio.open(path) as src:
        label = src.read(1).astype(np.int64)
        nodata = src.nodata
    if nodata is not None:
        label[label == int(nodata)] = -1
    if not zero_based:
        # Official dataset: 0 means no-data, 1..13 are crop/land-cover classes.
        valid = (label >= 1) & (label <= NUM_CLASSES)
        label = np.where(valid, label - 1, -1)
    else:
        label[~((label >= 0) & (label < NUM_CLASSES))] = -1
    return label


def make_synthetic(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create deterministic three-season HLS-like data and pseudo labels."""

    rng = np.random.default_rng(seed)
    height = width = TILE_SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    label = ((yy // 48) * 5 + (xx // 48)) % NUM_CLASSES
    label = label.astype(np.int64)
    label[((xx - 165) ** 2 + (yy - 60) ** 2) < 28**2] = 6  # water
    label[(yy > 150) & (np.abs(xx - 100) < 24)] = 1  # forest strip

    raw_by_time = np.empty((NUM_FRAMES, NUM_BANDS, height, width), np.float32)
    seasonal = np.asarray(
        [[0.90, 0.78, 0.64], [1.00, 1.08, 1.12], [1.10, 1.32, 0.88]],
        dtype=np.float32,
    )
    for t in range(NUM_FRAMES):
        for band in range(NUM_BANDS):
            base = BASE_MEANS[band]
            class_factor = 0.72 + 0.045 * label + 0.06 * np.sin(label + band)
            signal = base * class_factor * seasonal[t, band % 3]
            noise = rng.normal(0.0, BASE_STDS[band] * 0.18, size=(height, width))
            raw_by_time[t, band] = signal + noise
    # Distinct low-NIR/SWIR open-water signature at all three times.
    water = label == 6
    water_dn = np.asarray([350, 520, 410, 260, 180, 120], np.float32)
    for t in range(NUM_FRAMES):
        for band in range(NUM_BANDS):
            raw_by_time[t, band, water] = water_dn[band] + rng.normal(
                0.0, 35.0, int(water.sum())
            )
    raw = np.clip(raw_by_time.reshape(NUM_INPUT_CHANNELS, height, width), 0, 10000)
    label[:8, :8] = -1
    raw[:, :8, :8] = 0.0
    return raw.astype(np.float32), label


def preprocess(raw: np.ndarray) -> Tensor:
    if raw.ndim != 3 or raw.shape[0] != NUM_INPUT_CHANNELS:
        raise ValueError(f"Expected raw input shape (18,H,W), got {raw.shape}.")
    raw = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    normalized = (raw - MEANS_18[:, None, None]) / STDS_18[:, None, None]
    # Exact official pipeline: flat 18 channels -> (6 bands, 3 frames, H, W).
    return torch.from_numpy(
        np.ascontiguousarray(normalized.reshape(NUM_BANDS, NUM_FRAMES, *raw.shape[1:]))
    ).float()


def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.inference_mode()
def predict_sliding(
    model: PrithviCropSegmenter,
    normalized: Tensor,
    device: torch.device,
    stride: int,
) -> Tensor:
    if not 1 <= stride <= TILE_SIZE:
        raise ValueError("--stride must be between 1 and 224.")
    _, _, original_h, original_w = normalized.shape
    padded_h = max(original_h, TILE_SIZE)
    padded_w = max(original_w, TILE_SIZE)
    padded = F.pad(normalized, (0, padded_w - original_w, 0, padded_h - original_h))
    ys = tile_starts(padded_h, TILE_SIZE, stride)
    xs = tile_starts(padded_w, TILE_SIZE, stride)
    logits_sum = torch.zeros((NUM_CLASSES, padded_h, padded_w), device=device)
    counts = torch.zeros((1, padded_h, padded_w), device=device)
    padded = padded.to(device)
    for y in ys:
        for x in xs:
            tile = padded[:, :, y : y + TILE_SIZE, x : x + TILE_SIZE].unsqueeze(0)
            logits = model(tile)[0]
            logits_sum[:, y : y + TILE_SIZE, x : x + TILE_SIZE] += logits
            counts[:, y : y + TILE_SIZE, x : x + TILE_SIZE] += 1
    logits_sum /= counts
    return logits_sum[:, :original_h, :original_w]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    model: PrithviCropSegmenter,
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
        raise ValueError(f"Prediction/label shapes differ: {prediction.shape} vs {label.shape}.")
    valid = (label >= 0) & (label < NUM_CLASSES)
    if not valid.any():
        raise ValueError("Label has no valid class pixels.")
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (label[valid], prediction[valid]), 1)
    tp = np.diag(matrix).astype(np.float64)
    union = matrix.sum(1) + matrix.sum(0) - tp
    truth_count = matrix.sum(1)
    iou = np.divide(tp, union, out=np.full(NUM_CLASSES, np.nan), where=union > 0)
    accuracy = np.divide(
        tp, truth_count, out=np.full(NUM_CLASSES, np.nan), where=truth_count > 0
    )
    return {
        "valid_pixels": int(valid.sum()),
        "ignored_pixels": int(label.size - valid.sum()),
        "confusion_matrix_rows_truth_cols_prediction": matrix.tolist(),
        "class_iou": {CLASS_NAMES[i]: float(iou[i]) for i in range(NUM_CLASSES)},
        "class_accuracy": {
            CLASS_NAMES[i]: float(accuracy[i]) for i in range(NUM_CLASSES)
        },
        "overall_accuracy": float(tp.sum() / valid.sum()),
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
    output_profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write((prediction + 1).astype(np.uint8), 1)
    return str(path.resolve())


def rgb_for_time(raw: np.ndarray, time_index: int) -> np.ndarray:
    # Model-card GeoTIFF convention: six-band blocks repeated over three times.
    start = time_index * NUM_BANDS
    rgb = np.moveaxis(raw[[start + 2, start + 1, start]], 0, -1)
    finite = rgb[np.isfinite(rgb)]
    lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
    return np.clip((rgb - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def save_plot(
    path: Path,
    raw: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    label: Optional[np.ndarray],
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    columns = 5 if label is not None else 4
    figure, axes = plt.subplots(1, columns, figsize=(4.0 * columns, 4.0))
    axes[0].imshow(rgb_for_time(raw, 0))
    axes[0].set_title("Early-season RGB")
    axes[1].imshow(rgb_for_time(raw, 2))
    axes[1].set_title("Late-season RGB")
    axes[2].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Prediction confidence")
    axes[3].imshow(prediction, cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1)
    axes[3].set_title("13-class prediction")
    if label is not None:
        shown = np.ma.masked_where(label < 0, label)
        axes[4].imshow(shown, cmap="tab20", vmin=0, vmax=NUM_CLASSES - 1)
        axes[4].set_title("Label / synthetic target")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(path.resolve())


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
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
        "Prithvi multi-temporal crop-classification validation",
        "=" * 53,
        f"Status: {results['status']}",
        f"Validation level: {results['validation_level']}",
        f"Data source: {results['data_source']}",
        f"Device: {results['environment']['device']}",
        f"Input shape: {tuple(results['input']['shape'])}",
        f"Model input shape: {tuple(results['input']['model_tensor_shape'])}",
        f"Output shape: {tuple(results['output']['shape'])}",
        f"Strict checkpoint load: {results['model']['strict_load']}",
        f"Finite output: {results['output']['finite_fraction']:.6f}",
        f"Median latency: {results['performance']['median_latency_ms']:.3f} ms",
        f"Peak GPU memory: {results['performance']['peak_gpu_memory_mb']:.2f} MB",
        f"Mean confidence: {results['output']['mean_confidence']:.6f}",
    ]
    if metrics:
        qualifier = (
            "synthetic pseudo-label; not scientific model accuracy"
            if results["data_source"].startswith("synthetic")
            else "paired ground-truth label"
        )
        lines.extend(
            [
                f"Metric basis: {qualifier}",
                f"Overall accuracy: {metrics['overall_accuracy']:.6f}",
                f"Mean IoU: {metrics['mean_iou']:.6f}",
                f"Mean accuracy: {metrics['mean_accuracy']:.6f}",
            ]
        )
    else:
        lines.append("Scientific accuracy: unavailable (no paired label)")
    if results["data_source"].startswith("synthetic"):
        lines.append("Scientific accuracy: unavailable (synthetic pseudo-label only)")
    lines.append(f"Results: {results['output_dir']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_dir = args.model_dir.resolve()
    weights_path = args.weights.resolve() if args.weights else model_dir / WEIGHT_NAME
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

    if args.input and args.input.lower() != "none":
        input_path: Optional[Path] = Path(args.input).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input not found: {input_path}")
    elif args.input and args.input.lower() == "none":
        input_path = None
    else:
        input_path = find_local_input(model_dir, output_dir)

    profile: Optional[Dict[str, Any]] = None
    band_indexes = tuple(range(NUM_INPUT_CHANNELS))
    label_path: Optional[Path] = None
    if input_path is None:
        raw, label = make_synthetic(args.seed)
        data_source = "synthetic_multitemporal_hls_like"
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
        label = (
            load_label(label_path, args.label_zero_based) if label_path else None
        )
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
    probabilities = logits.softmax(dim=0).numpy()
    prediction = probabilities.argmax(axis=0).astype(np.int16)
    confidence = probabilities.max(axis=0)
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
            confidence,
            label,
        )

    class_counts = np.bincount(prediction.ravel(), minlength=NUM_CLASSES)
    class_fractions = class_counts / prediction.size
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
            "model_tensor_shape": [1, NUM_BANDS, NUM_FRAMES, raw.shape[1], raw.shape[2]],
            "band_names_per_time": list(BAND_NAMES),
            "time_steps": NUM_FRAMES,
            "zero_based_source_band_indexes": list(band_indexes),
            "normalization_means_18": MEANS_18.tolist(),
            "normalization_stds_18": STDS_18.tolist(),
            "official_preprocess_order": "normalize_flat_18_then_reshape_6x3",
            "tile_size": TILE_SIZE,
            "stride": args.stride,
        },
        "output": {
            "shape": list(logits.shape),
            "classes": list(CLASS_NAMES),
            "finite_fraction": finite_fraction,
            "mean_confidence": float(confidence.mean()),
            "minimum_confidence": float(confidence.min()),
            "maximum_confidence": float(confidence.max()),
            "predicted_class_fractions": {
                CLASS_NAMES[i]: float(class_fractions[i]) for i in range(NUM_CLASSES)
            },
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
            "validation_overall_accuracy": 0.6064,
            "validation_mean_iou": 0.4269,
            "validation_mean_accuracy": 0.6406,
            "note": "Published reference only; not recomputed without the official paired validation data.",
        },
        "output_dir": str(output_dir.resolve()),
    }
    results = json_ready(results)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(output_dir / "summary.txt", results)

    print("\n========== Prithvi crop classification validation ==========")
    print(f"Status:              {results['status']}")
    print(f"Validation level:    {validation_level}")
    print(f"Data source:         {data_source}")
    print(f"Device:              {device}")
    print(f"Input shape:         {tuple(raw.shape)}")
    print(f"Model input shape:   {(1, NUM_BANDS, NUM_FRAMES, raw.shape[1], raw.shape[2])}")
    print(f"Output shape:        {tuple(logits.shape)}")
    print(f"Strict load:         {model_metadata['strict_load']}")
    print(f"Parameters:          {model_metadata['parameter_count']:,}")
    print(f"Finite output:       {finite_fraction:.6f}")
    print(f"Median latency:      {statistics.median(timings):.3f} ms")
    print(f"Peak GPU memory:     {peak_gpu_mb:.2f} MB")
    print(f"Mean confidence:     {confidence.mean():.6f}")
    top_classes = np.argsort(class_fractions)[::-1][:3]
    print(
        "Top predicted classes: "
        + ", ".join(
            f"{CLASS_NAMES[i]}={class_fractions[i]:.4f}" for i in top_classes
        )
    )
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
