"""AirfRANS data utilities for the MARIO reproduction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .vtk_xml import read_vtk_xml

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object


@dataclass
class AirfransCase:
    name: str
    coords: np.ndarray
    sdf: np.ndarray
    normals: np.ndarray
    bl_mask: np.ndarray
    target: np.ndarray
    surf: np.ndarray
    freestream: np.ndarray

    @property
    def decoder_input(self) -> np.ndarray:
        return np.concatenate([self.coords, self.sdf, self.normals, self.bl_mask], axis=-1).astype(np.float32)


def load_manifest(data_root: str | Path) -> dict[str, list[str]]:
    path = Path(data_root) / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_names(data_root: str | Path, split: str, limit: int | None = None) -> list[str]:
    manifest = load_manifest(data_root)
    if split not in manifest:
        raise KeyError(f"Unknown split {split!r}; available: {sorted(manifest)}")
    names = list(manifest[split])
    return names[:limit] if limit is not None else names


def freestream_from_case_name(case_name: str) -> np.ndarray:
    parts = case_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse freestream and AoA from case name: {case_name}")
    u_inf = float(parts[2])
    alpha = math.radians(float(parts[3]))
    return np.array([math.cos(alpha) * u_inf, math.sin(alpha) * u_inf], dtype=np.float32)


def case_uinf_angle(case_name: str) -> tuple[float, float]:
    parts = case_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse Uinf/AoA from case name: {case_name}")
    return float(parts[2]), float(parts[3])


def boundary_layer_mask(sdf: np.ndarray, tau: float = 0.02) -> np.ndarray:
    """Compute the paper's parabolic boundary-layer mask from SDF values.

    The paper defines the mask through a normalized inverse distance d_hat.
    The AirfRANS VTK files do not store that normalized field directly, so we
    use abs(sdf) normalized by the sample maximum as a reproducible surrogate.
    """

    abs_sdf = np.abs(sdf.astype(np.float32))
    max_dist = float(np.max(abs_sdf))
    if max_dist <= 0.0:
        return np.ones_like(abs_sdf, dtype=np.float32)
    d_hat = 1.0 - np.clip(abs_sdf / max_dist, 0.0, 1.0)
    return np.maximum(0.0, (d_hat - (1.0 - tau)) / tau) ** 2


def _case_file(data_root: Path, case_name: str, suffix: str) -> Path:
    path = data_root / case_name / f"{case_name}{suffix}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _nearest_values(
    source_points: np.ndarray,
    target_points: np.ndarray,
    values: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> np.ndarray:
    source = np.asarray(source_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    vals = np.asarray(values, dtype=np.float32)
    out = np.empty((target.shape[0], vals.shape[1]), dtype=np.float32)
    for start in range(0, target.shape[0], chunk_size):
        stop = min(start + chunk_size, target.shape[0])
        diff = target[start:stop, None, :] - source[None, :, :]
        nearest = np.argmin(np.sum(diff * diff, axis=-1), axis=1)
        out[start:stop] = vals[nearest]
    return out


def _cache_path(cache_dir: str | Path | None, case_name: str) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"{case_name}.npz"


def _load_cached_case(path: Path) -> AirfransCase:
    with np.load(path) as data:
        return AirfransCase(
            name=str(data["name"].item()),
            coords=data["coords"].astype(np.float32),
            sdf=data["sdf"].astype(np.float32),
            normals=data["normals"].astype(np.float32),
            bl_mask=data["bl_mask"].astype(np.float32),
            target=data["target"].astype(np.float32),
            surf=data["surf"].astype(bool),
            freestream=data["freestream"].astype(np.float32),
        )


def _save_cached_case(path: Path, case: AirfransCase) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        name=np.array(case.name),
        coords=case.coords,
        sdf=case.sdf,
        normals=case.normals,
        bl_mask=case.bl_mask,
        target=case.target,
        surf=case.surf,
        freestream=case.freestream,
    )


def load_airfrans_case(
    data_root: str | Path,
    case_name: str,
    *,
    cache_dir: str | Path | None = None,
    tau: float = 0.02,
) -> AirfransCase:
    cache = _cache_path(cache_dir, case_name)
    if cache is not None and cache.exists():
        return _load_cached_case(cache)

    root = Path(data_root)
    internal_path = _case_file(root, case_name, "_internal.vtu")
    aerofoil_path = _case_file(root, case_name, "_aerofoil.vtp")
    internal = read_vtk_xml(
        internal_path,
        point_arrays=["U", "p", "nut", "implicit_distance"],
        read_points=True,
    )
    aerofoil = read_vtk_xml(aerofoil_path, point_arrays=["Normals"], read_points=True)
    if internal.points is None:
        raise ValueError(f"No Points array found in {internal_path}")
    if aerofoil.points is None or "Normals" not in aerofoil.point_data:
        raise ValueError(f"No aerofoil Points/Normals found in {aerofoil_path}")
    for key in ["U", "p", "nut", "implicit_distance"]:
        if key not in internal.point_data:
            raise ValueError(f"No point array {key!r} found in {internal_path}")

    coords = internal.points[:, :2].astype(np.float32)
    velocity = internal.point_data["U"][:, :2].astype(np.float32)
    pressure = internal.point_data["p"].reshape(-1, 1).astype(np.float32)
    nut = internal.point_data["nut"].reshape(-1, 1).astype(np.float32)
    sdf = -internal.point_data["implicit_distance"].reshape(-1, 1).astype(np.float32)
    surf = np.isclose(velocity[:, 0], 0.0)
    normals = np.zeros((coords.shape[0], 2), dtype=np.float32)
    if np.any(surf):
        normals[surf] = _nearest_values(
            aerofoil.points[:, :2].astype(np.float32),
            coords[surf],
            -aerofoil.point_data["Normals"][:, :2].astype(np.float32),
        )
    target = np.concatenate([velocity, pressure, nut], axis=-1).astype(np.float32)
    case = AirfransCase(
        name=case_name,
        coords=coords,
        sdf=sdf,
        normals=normals,
        bl_mask=boundary_layer_mask(sdf, tau=tau).astype(np.float32),
        target=target,
        surf=surf,
        freestream=freestream_from_case_name(case_name),
    )
    if cache is not None:
        _save_cached_case(cache, case)
    return case


def deterministic_indices(name: str, total: int, count: int | None, *, salt: str = "") -> np.ndarray:
    if count is None or total <= count:
        return np.arange(total, dtype=np.int64)
    digest = hashlib.sha256(f"{name}:{salt}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False))


def random_indices(total: int, count: int | None, rng: np.random.Generator) -> np.ndarray:
    if count is None or total <= count:
        return np.arange(total, dtype=np.int64)
    return rng.choice(total, size=count, replace=False)


def compute_output_stats(
    data_root: str | Path,
    split: str,
    *,
    stats_path: str | Path,
    cache_dir: str | Path | None = None,
    limit: int | None = None,
    tau: float = 0.02,
    log_every: int = 25,
) -> dict[str, np.ndarray]:
    names = split_names(data_root, split, limit)
    sum_y = sumsq_y = sum_c = sumsq_c = None
    count = 0
    case_count = 0
    for index, name in enumerate(names, start=1):
        case = load_airfrans_case(data_root, name, cache_dir=cache_dir, tau=tau)
        y = case.target.astype(np.float64)
        c = case.freestream.astype(np.float64)
        sum_y = y.sum(axis=0) if sum_y is None else sum_y + y.sum(axis=0)
        sumsq_y = (y * y).sum(axis=0) if sumsq_y is None else sumsq_y + (y * y).sum(axis=0)
        sum_c = c if sum_c is None else sum_c + c
        sumsq_c = c * c if sumsq_c is None else sumsq_c + c * c
        count += y.shape[0]
        case_count += 1
        if log_every and (index == 1 or index % log_every == 0 or index == len(names)):
            print(f"stats split={split} cases={index}/{len(names)} points={count}", flush=True)
    if count == 0 or sum_y is None or sumsq_y is None or sum_c is None or sumsq_c is None:
        raise ValueError(f"No cases found for split {split}")
    target_mean = (sum_y / count).astype(np.float32)
    target_std = np.sqrt(np.maximum(sumsq_y / count - target_mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    cond_mean = (sum_c / case_count).astype(np.float32)
    cond_std = np.sqrt(np.maximum(sumsq_c / case_count - cond_mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    stats = {
        "target_mean": target_mean,
        "target_std": target_std,
        "condition_mean": cond_mean,
        "condition_std": cond_std,
    }
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)
    return stats


def load_stats(stats_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(stats_path) as data:
        return {
            "target_mean": data["target_mean"].astype(np.float32),
            "target_std": data["target_std"].astype(np.float32),
            "condition_mean": data["condition_mean"].astype(np.float32),
            "condition_std": data["condition_std"].astype(np.float32),
        }


class AirfransCaseDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        cache_dir: str | Path | None = None,
        points_per_case: int | None = 16000,
        limit: int | None = None,
        stats: dict[str, np.ndarray] | None = None,
        normalize_condition: bool = True,
        deterministic: bool = False,
        tau: float = 0.02,
        base_seed: int = 42,
        preload: bool = False,
        names: list[str] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for AirfransCaseDataset")
        self.data_root = Path(data_root)
        if names is None:
            self.names = split_names(data_root, split, limit)
        else:
            selected = list(names)
            self.names = selected[:limit] if limit is not None else selected
        self.cache_dir = cache_dir
        self.points_per_case = points_per_case
        self.stats = stats
        self.normalize_condition = normalize_condition
        self.deterministic = deterministic
        self.tau = tau
        self.base_seed = base_seed
        self.epoch = 0
        self.preloaded_cases: dict[str, AirfransCase] | None = None
        if preload:
            self.preloaded_cases = {
                name: load_airfrans_case(self.data_root, name, cache_dir=self.cache_dir, tau=self.tau)
                for name in self.names
            }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.names[index]
        if self.preloaded_cases is not None:
            case = self.preloaded_cases[name]
        else:
            case = load_airfrans_case(self.data_root, name, cache_dir=self.cache_dir, tau=self.tau)
        if self.deterministic:
            idx = deterministic_indices(name, case.coords.shape[0], self.points_per_case, salt=str(self.epoch))
        else:
            seed = self.base_seed + self.epoch * max(len(self.names), 1) + index
            rng = np.random.default_rng(seed)
            idx = random_indices(case.coords.shape[0], self.points_per_case, rng)

        target = case.target[idx]
        if self.stats is not None:
            target = (target - self.stats["target_mean"]) / (self.stats["target_std"] + 1e-8)
        condition = case.freestream
        if self.stats is not None and self.normalize_condition:
            condition = (condition - self.stats["condition_mean"]) / (self.stats["condition_std"] + 1e-8)

        return {
            "name": name,
            "decoder_x": torch.from_numpy(case.decoder_input[idx].astype(np.float32)),
            "coords": torch.from_numpy(case.coords[idx].astype(np.float32)),
            "sdf": torch.from_numpy(case.sdf[idx].astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "condition": torch.from_numpy(condition.astype(np.float32)),
            "surf": torch.from_numpy(case.surf[idx]),
            "mask": torch.ones(len(idx), dtype=torch.float32),
        }


def collate_cases(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for collate_cases")
    max_points = max(item["decoder_x"].shape[0] for item in batch)
    decoder_dim = batch[0]["decoder_x"].shape[-1]
    target_dim = batch[0]["target"].shape[-1]
    coords = torch.zeros(len(batch), max_points, 2, dtype=batch[0]["coords"].dtype)
    sdf = torch.zeros(len(batch), max_points, 1, dtype=batch[0]["sdf"].dtype)
    decoder_x = torch.zeros(len(batch), max_points, decoder_dim, dtype=batch[0]["decoder_x"].dtype)
    target = torch.zeros(len(batch), max_points, target_dim, dtype=batch[0]["target"].dtype)
    surf = torch.zeros(len(batch), max_points, dtype=torch.bool)
    mask = torch.zeros(len(batch), max_points, dtype=batch[0]["mask"].dtype)
    condition = torch.stack([item["condition"] for item in batch], dim=0)
    names: list[str] = []
    for row, item in enumerate(batch):
        n = item["decoder_x"].shape[0]
        coords[row, :n] = item["coords"]
        sdf[row, :n] = item["sdf"]
        decoder_x[row, :n] = item["decoder_x"]
        target[row, :n] = item["target"]
        surf[row, :n] = item["surf"]
        mask[row, :n] = item["mask"]
        names.append(str(item["name"]))
    return {
        "name": names,
        "coords": coords,
        "sdf": sdf,
        "decoder_x": decoder_x,
        "target": target,
        "condition": condition,
        "surf": surf,
        "mask": mask,
    }
