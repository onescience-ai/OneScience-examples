#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NowcastNet-Rewritten 一键复现脚本（从头开始版）

适用场景
--------
当前目录至少需要：
- reproduce_nowcastnet.py

可选文件：
- README.md
- LICENSE
- gitattributes 或 .gitattributes

本脚本会自动完成：
1. 下载 rewritten_model.pt 模型权重（若本地不存在）；
2. 下载完整源码 nowcastnet-rewritten（若本地不存在）；
3. 生成模型所需的 29 张 512×512 16位灰度 PNG 输入图像；
4. 安装/检查依赖；
5. 使用原生 RadarDataset 读取数据；
6. 检查权重；
7. 执行模型推理；
8. 输出预测结果与测试指标；
9. 生成 metrics JSON 报告。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
import zlib
from array import array
from pathlib import Path
from typing import Iterable, Optional

REPO_URL = (
    "https://github.com/VioletsOleander/"
    "nowcastnet-rewritten/archive/refs/heads/main.zip"
)
REPO_DIR_NAME = "nowcastnet-rewritten"
WEIGHTS_NAME = "rewritten_model.pt"
WEIGHTS_URL = (
    "https://huggingface.co/VioletsOleander/"
    "nowcastnet-rewritten/resolve/main/rewritten_model.pt?download=true"
)
RUNTIME_DIR_NAME = "runtime_data"
DATASET_SUBDIR = Path("synthetic_radar")
CASE_NAME = "synthetic_case_001"
RESULTS_DIR_NAME = "nowcastnet_results"
CONFIG_NAME = "inference.auto.toml"
LOG_NAME = "inference_nowcastnet.log"
METRICS_NAME = "nowcastnet_metrics.json"
REQUIRED_PYTHON = (3, 10)
WIDTH = 512
HEIGHT = 512
TOTAL_FRAMES = 29
INPUT_LENGTH = 9
PRED_LENGTH = 20
DEFAULT_SEED = 42

class ReproductionError(RuntimeError):
    pass

def print_header(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}", flush=True)

def print_step(index: int, total: int, title: str) -> None:
    print(f"\n[{index}/{total}] {title}", flush=True)

def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"

def run_command(command: list[str], *, cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    shown = " ".join(str(x) for x in command)
    print(f"$ {shown}", flush=True)
    kwargs = {"cwd": str(cwd) if cwd else None, "env": env, "text": True, "check": False}
    if capture:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
        if result.stdout:
            print(result.stdout, end="", flush=True)
    else:
        result = subprocess.run(command, **kwargs)
    if check and result.returncode != 0:
        raise ReproductionError(f"命令执行失败，退出码为 {result.returncode}：\n{shown}")
    return result

def pythonpath_env(repo_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_dir) if not current else str(repo_dir) + os.pathsep + current
    return env

def module_import_test(module_name: str, *, repo_dir: Optional[Path] = None) -> bool:
    env = os.environ.copy()
    if repo_dir is not None:
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(repo_dir) if not current else str(repo_dir) + os.pathsep + current
    result = subprocess.run([sys.executable, "-c", f"import {module_name}"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0

def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ReproductionError(f"ZIP 中存在不安全路径：{member.filename}") from exc
        archive.extractall(destination)

def verify_base_files(base_dir: Path) -> Path:
    weights_path = base_dir / WEIGHTS_NAME
    print(f"脚本目录：{base_dir}")

    if not weights_path.is_file():
        print(f"当前目录缺少模型权重，将自动下载：{weights_path}")
        download_file(
            WEIGHTS_URL,
            weights_path,
            minimum_size=50 * 1024 * 1024,
            description="NowcastNet 模型权重",
        )

    if weights_path.stat().st_size < 50 * 1024 * 1024:
        print("检测到权重文件过小，将删除后重新下载。")
        weights_path.unlink(missing_ok=True)
        download_file(
            WEIGHTS_URL,
            weights_path,
            minimum_size=50 * 1024 * 1024,
            description="NowcastNet 模型权重",
        )

    print(
        f"  [存在] {WEIGHTS_NAME:25s} "
        f"{human_size(weights_path.stat().st_size)}"
    )

    for optional_name in (
        "README.md",
        "LICENSE",
        "gitattributes",
        ".gitattributes",
    ):
        path = base_dir / optional_name
        if path.exists():
            size = (
                human_size(path.stat().st_size)
                if path.is_file()
                else "<DIR>"
            )
            print(f"  [存在] {optional_name:25s} {size}")

    return weights_path

def check_environment() -> dict[str, object]:
    info = {"python_executable": sys.executable, "python_version": platform.python_version(), "platform": platform.platform()}
    if sys.version_info[:2] != REQUIRED_PYTHON:
        raise ReproductionError(f"当前 Python 为 {platform.python_version()}。\nNowcastNet-Rewritten 推荐使用 Python 3.10。")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info

def download_file(url: str, destination: Path, *, minimum_size: int, description: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 NowcastNetReproduction"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            downloaded = 0
            last_print = 0.0
            with temp_path.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    downloaded += len(block)
                    now = time.monotonic()
                    if now - last_print >= 1.0:
                        if total:
                            percent = downloaded / total * 100
                            print(f"\r正在下载{description}：{human_size(downloaded)} / {human_size(total)} ({percent:.1f}%)", end="", flush=True)
                        else:
                            print(f"\r正在下载{description}：{human_size(downloaded)}", end="", flush=True)
                        last_print = now
        print()
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise ReproductionError(f"自动下载{description}失败：{url}\n请检查网络连接。") from exc
    if not temp_path.is_file() or temp_path.stat().st_size < minimum_size:
        actual = temp_path.stat().st_size if temp_path.exists() else 0
        temp_path.unlink(missing_ok=True)
        raise ReproductionError(f"{description}下载不完整：仅 {human_size(actual)}。")
    temp_path.replace(destination)
    print(f"{description}下载完成：{destination}")
    return destination

def ensure_source_code(base_dir: Path) -> Path:
    repo_dir = base_dir / REPO_DIR_NAME
    required = [repo_dir / "pyproject.toml", repo_dir / "nowcastnet" / "__init__.py", repo_dir / "nowcastnet" / "inference.py", repo_dir / "nowcastnet" / "datasets" / "radar_dataset.py"]
    if all(path.is_file() for path in required):
        print(f"完整源码已存在：{repo_dir}")
        return repo_dir
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    zip_path = base_dir / "nowcastnet-rewritten-main.zip"
    download_file(REPO_URL, zip_path, minimum_size=10_000, description="NowcastNet 源代码")
    with tempfile.TemporaryDirectory(prefix="nowcastnet_extract_", dir=base_dir) as temp_dir:
        temp_root = Path(temp_dir)
        safe_extract_zip(zip_path, temp_root)
        candidates = [path for path in temp_root.iterdir() if path.is_dir() and (path / "pyproject.toml").is_file()]
        if len(candidates) != 1:
            raise ReproductionError("无法识别下载后的源码根目录。")
        shutil.move(str(candidates[0]), str(repo_dir))
    zip_path.unlink(missing_ok=True)
    if not all(path.is_file() for path in required):
        raise ReproductionError("源码下载后仍缺少关键文件。")
    print(f"源码准备完成：{repo_dir}")
    return repo_dir


def patch_source_compatibility(repo_dir: Path) -> Path:
    """Patch upstream RadarDataset for OpenCV builds requiring str paths."""
    radar_dataset_path = (
        repo_dir / "nowcastnet" / "datasets" / "radar_dataset.py"
    )
    if not radar_dataset_path.is_file():
        raise ReproductionError(
            f"缺少 RadarDataset 源文件：{radar_dataset_path}"
        )

    text = radar_dataset_path.read_text(encoding="utf-8")
    old = "cv.imread(frame_path, cv.IMREAD_UNCHANGED)"
    new = "cv.imread(str(frame_path), cv.IMREAD_UNCHANGED)"

    if new in text:
        print("OpenCV 路径兼容修复已存在，跳过重复修改。")
        return radar_dataset_path

    if old not in text:
        raise ReproductionError(
            "未找到预期的 cv.imread(frame_path, ...) 代码，"
            "上游源码可能已更新，请人工检查：\n"
            f"  {radar_dataset_path}"
        )

    backup_path = radar_dataset_path.with_suffix(".py.orig")
    if not backup_path.exists():
        backup_path.write_text(text, encoding="utf-8")

    patched = text.replace(old, new)
    radar_dataset_path.write_text(patched, encoding="utf-8")
    print(
        "已应用 OpenCV 路径兼容修复："
        "cv.imread(frame_path, ...) -> cv.imread(str(frame_path), ...)"
    )
    return radar_dataset_path

def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

def write_grayscale_png16(path: Path, rows: Iterable[Iterable[int]], width: int, height: int) -> None:
    raw = bytearray()
    row_count = 0
    for row in rows:
        values = array("H", row)
        if len(values) != width:
            raise ReproductionError(f"第 {row_count} 行像素数错误：{len(values)} != {width}")
        if sys.byteorder == "little":
            values.byteswap()
        raw.append(0)
        raw.extend(values.tobytes())
        row_count += 1
    if row_count != height:
        raise ReproductionError(f"图像行数错误：{row_count} != {height}")
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    payload = signature + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6)) + png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)

def compact_blob(x: int, y: int, center_x: float, center_y: float, radius_x: float, radius_y: float, amplitude: float) -> float:
    dx = (x - center_x) / radius_x
    dy = (y - center_y) / radius_y
    q = dx * dx + dy * dy
    if q >= 1.0:
        return 0.0
    shape = 1.0 - q
    return amplitude * shape * shape

def rain_band(x: int, y: int, frame_index: int) -> float:
    slope = 0.26 + 0.003 * frame_index
    center_y = 252.0 - 0.45 * frame_index
    distance = abs((y - center_y) - slope * (x - 256))
    half_width = 18.0 + 0.08 * frame_index
    if distance >= half_width:
        return 0.0
    along = abs(x - (250 + frame_index))
    if along >= 190:
        return 0.0
    cross_shape = 1.0 - distance / half_width
    along_shape = 1.0 - along / 190.0
    return 14.0 * cross_shape * cross_shape * along_shape

def deterministic_noise(x: int, y: int, frame_index: int, seed: int) -> float:
    value = (x * 73856093 ^ y * 19349663 ^ frame_index * 83492791 ^ seed * 2654435761) & 0xFFFFFFFF
    return ((value % 17) - 8) * 0.035

def rainfall_value(x: int, y: int, frame_index: int, seed: int) -> float:
    amp1 = 50.0 + 2.2 * frame_index if frame_index <= 12 else 76.4 - 1.8 * (frame_index - 12)
    cell1 = compact_blob(x, y, 122.0 + 5.0 * frame_index, 118.0 + 3.2 * frame_index, 42.0 + 0.35 * frame_index, 34.0 + 0.25 * frame_index, max(24.0, amp1))
    amp2 = 27.0 + 7.0 * math.sin((frame_index + 2) / 5.0)
    cell2 = compact_blob(x, y, 328.0 + 2.1 * frame_index, 342.0 - 2.3 * frame_index, 68.0, 48.0, amp2)
    amp3 = min(max(0.0, (frame_index - 7) * 2.6), 39.0)
    cell3 = compact_blob(x, y, 410.0 - 2.9 * frame_index, 104.0 + 2.0 * frame_index, 29.0 + 0.15 * frame_index, 25.0 + 0.15 * frame_index, amp3)
    value = cell1 + cell2 + cell3 + rain_band(x, y, frame_index)
    value += deterministic_noise(x, y, frame_index, seed)
    return min(100.0, max(0.0, value))

def encoded_rows(frame_index: int, seed: int) -> Iterable[list[int]]:
    for y in range(HEIGHT):
        row = []
        for x in range(WIDTH):
            rainfall = rainfall_value(x, y, frame_index, seed)
            pixel_value = round((rainfall + 3.0) * 10.0)
            row.append(min(65535, max(0, pixel_value)))
        yield row

def expected_frame_paths(dataset_root: Path) -> list[Path]:
    case_dir = dataset_root / CASE_NAME
    return [case_dir / f"{CASE_NAME}-{index:02d}.png" for index in range(TOTAL_FRAMES)]

def frames_are_complete(dataset_root: Path) -> bool:
    paths = expected_frame_paths(dataset_root)
    return all(path.is_file() and path.stat().st_size > 100 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for path in paths)

def generate_input_images(base_dir: Path, *, force: bool, seed: int = DEFAULT_SEED) -> tuple[Path, dict[str, object]]:
    dataset_root = base_dir / RUNTIME_DIR_NAME / DATASET_SUBDIR
    case_dir = dataset_root / CASE_NAME
    if frames_are_complete(dataset_root) and not force:
        print(f"仿真输入已存在，跳过生成：{case_dir}")
    else:
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        for idx, frame_path in enumerate(expected_frame_paths(dataset_root)):
            frame_start = time.perf_counter()
            write_grayscale_png16(frame_path, encoded_rows(idx, seed), WIDTH, HEIGHT)
            elapsed = time.perf_counter() - frame_start
            print(f"[{idx + 1:02d}/{TOTAL_FRAMES}] 已生成 {frame_path.name}，耗时 {elapsed:.2f} 秒", flush=True)
        total_elapsed = time.perf_counter() - start
        print(f"输入图像生成总耗时：{total_elapsed:.2f} 秒")
    if not frames_are_complete(dataset_root):
        raise ReproductionError("输入图像生成后仍有文件缺失。")
    frame_paths = expected_frame_paths(dataset_root)
    sizes = [path.stat().st_size for path in frame_paths]
    info = {"dataset_root": str(dataset_root.resolve()), "case_name": CASE_NAME, "frame_count": TOTAL_FRAMES, "input_length": INPUT_LENGTH, "pred_length": PRED_LENGTH, "width": WIDTH, "height": HEIGHT, "bit_depth": 16, "seed": seed, "first_frame": frame_paths[0].name, "last_frame": frame_paths[-1].name, "total_size_bytes": sum(sizes), "encoding": "pixel_value = round((rainfall + 3) * 10)"}
    manifest_path = dataset_root.parent / "synthetic_manifest.json"
    manifest_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return dataset_root, info

def install_dependencies(*, reinstall: bool) -> None:
    python = sys.executable
    print(f"当前 Python：{python}")
    print(f"Python 版本：{platform.python_version()}")
    print(f"操作系统：{platform.platform()}")
    if reinstall or not module_import_test("pkg_resources"):
        run_command([python, "-m", "pip", "install", "--disable-pip-version-check", "setuptools==80.9.0"])
    requirements = {"numpy": "numpy>=2.2.2", "matplotlib": "matplotlib>=3.10.1", "cv2": "opencv-python-headless>=4.11.0.86", "scikit_learn": "scikit-learn>=1.6.1", "tomlkit": "tomlkit>=0.13.2", "pysteps": "pysteps>=1.14.0,<=1.18.0"}
    missing_specs = []
    for module_name, package_spec in requirements.items():
        actual_module = "sklearn" if module_name == "scikit_learn" else module_name
        if reinstall or not module_import_test(actual_module):
            missing_specs.append(package_spec)
    if missing_specs:
        run_command([python, "-m", "pip", "install", "--disable-pip-version-check", *missing_specs])
    else:
        print("核心依赖已满足，跳过重复安装。")
    if reinstall or not module_import_test("torch"):
        run_command([python, "-m", "pip", "install", "--disable-pip-version-check", "torch>=2.0.0"])
    required_modules = ["pkg_resources", "numpy", "matplotlib", "cv2", "sklearn", "tomlkit", "pysteps", "torch"]
    failed = [name for name in required_modules if not module_import_test(name)]
    if failed:
        raise ReproductionError("依赖安装后，以下模块仍无法导入：\n" + "\n".join(f"  - {name}" for name in failed))
    print("依赖导入检查全部通过。")

def validate_generated_pngs(dataset_root: Path) -> dict[str, object]:
    import cv2, numpy as np
    case_dir = dataset_root / CASE_NAME
    frame_paths = [case_dir / f"{CASE_NAME}-{index:02d}.png" for index in range(TOTAL_FRAMES)]
    shapes, dtypes = set(), set()
    decoded_min, decoded_max = float("inf"), float("-inf")
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise ReproductionError(f"OpenCV 无法读取：{frame_path}")
        shapes.add(tuple(frame.shape))
        dtypes.add(str(frame.dtype))
        decoded = np.clip(frame.astype(np.float32) / 10.0 - 3.0, 0, 128)
        decoded_min = min(decoded_min, float(decoded.min()))
        decoded_max = max(decoded_max, float(decoded.max()))
    if shapes != {(512, 512)}:
        raise ReproductionError(f"生成图片尺寸错误：{shapes}")
    info = {"frame_count": TOTAL_FRAMES, "image_shapes": [list(shape) for shape in sorted(shapes)], "image_dtypes": sorted(dtypes), "decoded_min": decoded_min, "decoded_max": decoded_max, "input_frames": INPUT_LENGTH, "target_frames": PRED_LENGTH}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info

def test_dataset_loader(repo_dir: Path, dataset_root: Path) -> dict[str, object]:
    code = f"""
import json
from nowcastnet.datasets.radar_dataset import RadarDataset
config = {{
    \"input_data_type\": \"float32\",\n    \"output_data_type\": \"float32\",\n    \"image_width\": 512,\n    \"image_height\": 512,\n    \"pred_length\": 20,\n    \"input_length\": 9,\n    \"dataset_path\": {str(dataset_root)!r},\n}}
dataset = RadarDataset(config)
observed, target = dataset[0]
result = {{
    \"dataset_length\": len(dataset),\n    \"observed_shape\": list(observed.shape),\n    \"target_shape\": list(target.shape),\n    \"observed_dtype\": str(observed.dtype),\n    \"target_dtype\": str(target.dtype),\n    \"observed_min\": float(observed.min()),\n    \"observed_max\": float(observed.max()),\n    \"target_min\": float(target.min()),\n    \"target_max\": float(target.max()),\n}}
print(\"DATASET_TEST_JSON=\" + json.dumps(result))
"""
    result = run_command([sys.executable, "-c", code], cwd=repo_dir, env=pythonpath_env(repo_dir), capture=True)
    marker = "DATASET_TEST_JSON="
    lines = [line for line in (result.stdout or "").splitlines() if line.startswith(marker)]
    if not lines:
        raise ReproductionError("没有获得 RadarDataset 测试结果。")
    info = json.loads(lines[-1][len(marker):])
    if info["observed_shape"] != [9, 512, 512]:
        raise ReproductionError(f"历史输入形状错误：{info['observed_shape']}")
    if info["target_shape"] != [20, 512, 512]:
        raise ReproductionError(f"未来标签形状错误：{info['target_shape']}")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info

def inspect_runtime(force_cpu: bool) -> dict[str, object]:
    import torch
    cuda_available = bool(torch.cuda.is_available())
    device = "cpu" if force_cpu or not cuda_available else "cuda:0"
    info = {"torch_version": torch.__version__, "cuda_available": cuda_available, "selected_device": device}
    if cuda_available:
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info

def inspect_weight_file(weights_path: Path, device: str) -> dict[str, object]:
    import torch
    state = torch.load(weights_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ReproductionError(f"权重对象类型为 {type(state).__name__}，预期为 dict。")
    tensor_count = 0
    parameter_elements = 0
    first_keys = []
    for key, value in state.items():
        if len(first_keys) < 8:
            first_keys.append(str(key))
        if hasattr(value, "numel"):
            tensor_count += 1
            parameter_elements += int(value.numel())
    info = {"file_size": human_size(weights_path.stat().st_size), "state_dict_entries": len(state), "tensor_entries": tensor_count, "parameter_elements": parameter_elements, "first_keys": first_keys, "inference_device": device}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info

def write_config(base_dir: Path, dataset_root: Path, weights_path: Path, device: str) -> Path:
    config_path = base_dir / CONFIG_NAME
    results_path = base_dir / RESULTS_DIR_NAME
    log_path = base_dir / LOG_NAME
    def toml_path(path: Path) -> str:
        text = str(path.resolve())
        text = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{text}"'
    config = f"""# reproduce_nowcastnet.py 自动生成
dataset_path = {toml_path(dataset_root)}
weights_path = {toml_path(weights_path)}
results_path = {toml_path(results_path)}

dataset_name = \"radar\"
input_length = 9
pred_length = 20
image_height = 512
image_width = 512

cpu_workers = 0
case_type = \"normal\"
crop_size = 384
batch_size = 1

generator_base_channels = 32
device = \"{device}\"

save_original_data = true
log_path = {toml_path(log_path)}
seed = 42
"""
    config_path.write_text(config, encoding="utf-8")
    print(f"推理配置已生成：{config_path}")
    return config_path

def run_inference(repo_dir: Path, config_path: Path, base_dir: Path, initial_device: str) -> tuple[str, float]:
    command = [sys.executable, "-m", "nowcastnet.inference", "--config_path", str(config_path)]
    start = time.perf_counter()
    result = run_command(command, cwd=repo_dir, env=pythonpath_env(repo_dir), capture=True, check=False)
    elapsed = time.perf_counter() - start
    if result.returncode == 0:
        return initial_device, elapsed
    output = result.stdout or ""
    cuda_error = any(phrase in output.lower() for phrase in ("cuda", "cudnn", "out of memory", "no kernel image", "driver"))
    if initial_device.startswith("cuda") and cuda_error:
        print("CUDA 推理失败，自动改用 CPU 重试。")
        text = config_path.read_text(encoding="utf-8").replace(f'device = \"{initial_device}\"', 'device = \"cpu\"')
        config_path.write_text(text, encoding="utf-8")
        start = time.perf_counter()
        result = run_command(command, cwd=repo_dir, env=pythonpath_env(repo_dir), capture=True, check=False)
        elapsed = time.perf_counter() - start
        if result.returncode == 0:
            return "cpu", elapsed
    raise ReproductionError("NowcastNet 推理失败。请查看上方输出和日志：\n" + f"  {base_dir / LOG_NAME}")

def validate_outputs(base_dir: Path, inference_seconds: float) -> dict[str, object]:
    import numpy as np
    results_root = base_dir / RESULTS_DIR_NAME
    candidates = list(results_root.rglob("frames.npy"))
    if not candidates:
        raise ReproductionError(f"没有找到预测数组：{results_root}")
    array_path = candidates[0]
    result_dir = array_path.parent
    frames = np.load(array_path)
    png_files = sorted(result_dir.glob("*.png"))
    if frames.ndim != 3 or frames.shape[0] != 20:
        raise ReproductionError(f"预测数组形状异常：{frames.shape}")
    info = {"result_directory": str(result_dir), "frames_npy": str(array_path), "prediction_shape": list(frames.shape), "prediction_dtype": str(frames.dtype), "prediction_min": float(np.nanmin(frames)), "prediction_max": float(np.nanmax(frames)), "prediction_mean": float(np.nanmean(frames)), "prediction_png_count": len(png_files), "inference_seconds": round(inference_seconds, 4)}
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info

def write_metrics_report(base_dir: Path, *, environment: dict[str, object], generated_input: dict[str, object], png_validation: dict[str, object], dataset_test: dict[str, object], runtime: dict[str, object], weight_info: dict[str, object], output_info: dict[str, object], final_device: str) -> Path:
    report_path = base_dir / METRICS_NAME
    report = {"completed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "base_directory": str(base_dir), "environment": environment, "generated_input": generated_input, "png_validation": png_validation, "dataset_test": dataset_test, "runtime": runtime, "weight_info": weight_info, "output_info": output_info, "final_device": final_device, "conclusion": "输入图像生成、数据读取、权重检查、模型推理和输出验证均已完成。", "notice": "仿真输入仅用于验证程序流程，不能用于评价真实天气预报精度。"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NowcastNet-Rewritten 从头复现脚本")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用 CPU。")
    parser.add_argument("--rerun", action="store_true", help="删除旧预测结果并重新推理。")
    parser.add_argument("--regenerate-input", action="store_true", help="删除并重新生成29帧仿真输入。")
    parser.add_argument("--reinstall", action="store_true", help="重新安装依赖。")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    print_header("NowcastNet-Rewritten 一键复现")
    print("脚本会自动生成输入图像，并完成模型运行与测试指标输出。")
    total_steps = 10
    print_step(1, total_steps, "检查当前目录中的基础文件")
    weights_path = verify_base_files(base_dir)
    print_step(2, total_steps, "检查 Python 与系统环境")
    environment = check_environment()
    print_step(3, total_steps, "准备完整源码")
    repo_dir = ensure_source_code(base_dir)
    patch_source_compatibility(repo_dir)
    print_step(4, total_steps, "生成模型输入图像")
    dataset_root, generated_input = generate_input_images(base_dir, force=args.regenerate_input)
    print_step(5, total_steps, "安装并检查依赖")
    install_dependencies(reinstall=args.reinstall)
    print_step(6, total_steps, "验证生成的 PNG 输入图像")
    png_validation = validate_generated_pngs(dataset_root)
    print_step(7, total_steps, "使用原生 RadarDataset 读取数据")
    dataset_test = test_dataset_loader(repo_dir, dataset_root)
    print_step(8, total_steps, "检查运行设备与模型权重")
    runtime = inspect_runtime(force_cpu=args.force_cpu)
    selected_device = str(runtime["selected_device"])
    weight_info = inspect_weight_file(weights_path, selected_device)
    config_path = write_config(base_dir, dataset_root, weights_path, selected_device)
    results_dir = base_dir / RESULTS_DIR_NAME
    existing_frames = list(results_dir.rglob("frames.npy")) if results_dir.exists() else []
    if args.rerun and results_dir.exists():
        shutil.rmtree(results_dir)
        existing_frames = []
    print_step(9, total_steps, "执行模型推理")
    if existing_frames and not args.rerun:
        print(f"检测到已有预测结果：{existing_frames[0]}\n本次复用旧结果；需要重新推理时请添加 --rerun。")
        final_device = selected_device
        inference_seconds = 0.0
    else:
        final_device, inference_seconds = run_inference(repo_dir, config_path, base_dir, selected_device)
    print_step(10, total_steps, "验证输出并写出测试指标")
    output_info = validate_outputs(base_dir, inference_seconds)
    metrics_path = write_metrics_report(base_dir, environment=environment, generated_input=generated_input, png_validation=png_validation, dataset_test=dataset_test, runtime=runtime, weight_info=weight_info, output_info=output_info, final_device=final_device)
    print_header("复现成功")
    print(f"输入图像目录：{dataset_root}")
    print(f"预测结果目录：{output_info['result_directory']}")
    print(f"预测数组文件：{output_info['frames_npy']}")
    print(f"预测张量形状：{output_info['prediction_shape']}")
    print(f"测试指标文件：{metrics_path}")
    print(f"推理日志文件：{base_dir / LOG_NAME}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中止运行。", file=sys.stderr)
        raise SystemExit(130)
    except ReproductionError as exc:
        print_header("复现未完成")
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print_header("发生未预期错误")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
