#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import os
import sys
import platform
import shutil
import subprocess
import importlib
import importlib.util
import importlib.metadata as metadata
import json
import time
import gc

ROOT = Path.cwd().resolve()

print("Notebook 当前目录：")
print(ROOT)

assert (ROOT / "pyproject.toml").is_file(), (
    "当前目录没有 pyproject.toml。\n"
    "请关闭 Notebook，将它移动到 Swift 源码根目录后重新打开。"
)

assert (ROOT / "src" / "swift").is_dir(), (
    "没有找到 src/swift。\n"
    "说明 Notebook 不在正确的 Swift 源码根目录。"
)

print("\n源码根目录检查通过。")
print("pyproject.toml：", ROOT / "pyproject.toml")
print("Swift 包目录：", ROOT / "src" / "swift")


# In[2]:


import torch

def get_module_version(module_name, distribution_name=None):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if version is not None:
            return str(version)
    except Exception:
        pass

    try:
        return metadata.version(distribution_name or module_name)
    except Exception:
        return "未知"

def read_total_memory_gb():
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None

    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            kb = int(line.split()[1])
            return kb / 1024**2
    return None

print("=" * 70)
print("基础运行环境")
print("=" * 70)
print("Python 可执行文件：", sys.executable)
print("Python 版本：", sys.version.replace("\n", " "))
print("操作系统：", platform.platform())
print("机器架构：", platform.machine())
print("主机名：", platform.node())
print("CPU 信息：", platform.processor() or "未提供")

ram_gb = read_total_memory_gb()
if ram_gb is not None:
    print(f"系统内存：{ram_gb:.2f} GB")

disk = shutil.disk_usage(ROOT)
print(f"当前磁盘总量：{disk.total / 1024**3:.2f} GB")
print(f"当前磁盘可用：{disk.free / 1024**3:.2f} GB")

print("\nPyTorch 信息")
print("-" * 70)
print("torch 版本：", torch.__version__)
print("torch CUDA 编译版本：", torch.version.cuda)
print("torch CUDA 是否可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA 设备数量：", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"设备 {i}：{props.name}，"
            f"显存 {props.total_memory / 1024**3:.2f} GB"
        )

# 部分非 NVIDIA 后端会把设备接口注册到 torch.npu、torch.xpu、torch.musa 等
for backend_name in ["npu", "xpu", "musa", "mlu"]:
    backend = getattr(torch, backend_name, None)
    if backend is None:
        continue

    is_available = getattr(backend, "is_available", None)
    if callable(is_available):
        try:
            available = is_available()
        except Exception:
            available = False

        print(f"torch.{backend_name} 是否可用：", available)

        if available:
            count_fn = getattr(backend, "device_count", None)
            name_fn = getattr(backend, "get_device_name", None)

            if callable(count_fn):
                count = count_fn()
                print(f"{backend_name} 设备数量：", count)

                if callable(name_fn):
                    for i in range(count):
                        try:
                            print(f"设备 {i}：", name_fn(i))
                        except Exception:
                            pass

print("\n可能的镜像信息环境变量")
print("-" * 70)

image_keys = [
    "FLAGOS_IMAGE",
    "IMAGE_NAME",
    "JUPYTER_IMAGE_SPEC",
    "JUPYTERHUB_IMAGE",
    "CONDA_DEFAULT_ENV",
    "HOSTNAME",
]

IMAGE_INFO = {}

for key in image_keys:
    value = os.environ.get(key)
    if value:
        IMAGE_INFO[key] = value
        print(f"{key}={value}")

if not IMAGE_INFO:
    print("Notebook 内没有暴露明确镜像名。")
    print("请从 FlagOS/SCNET 创建 Notebook 的页面手动记录：")
    print("1. 所属组")
    print("2. 镜像完整名称")
    print("3. 镜像版本或标签")


# In[3]:


CORE_MODULES = ["torch", "triton", "flag_gems"]

core_results = {}

for name in CORE_MODULES:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "未提供 __version__")
        core_results[name] = {
            "status": "成功",
            "version": str(version),
        }
        print(f"[成功] import {name}")
        print(f"       版本：{version}")
    except Exception as exc:
        core_results[name] = {
            "status": "失败",
            "error": repr(exc),
        }
        print(f"[失败] import {name}")
        print(f"       错误：{exc!r}")

failed_core = [
    name for name, result in core_results.items()
    if result["status"] != "成功"
]

assert not failed_core, (
    f"FlagOS 镜像缺少核心组件：{failed_core}\n"
    "不要在这里直接 pip install 普通 torch 或普通 triton。\n"
    "请重新选择正确的 FlagOS/FlagGems 镜像。"
)

import triton
import flag_gems

print("\nFlagGems 设备：", getattr(flag_gems, "device", "未提供"))
print("\n核心环境检查全部通过。")
print("注意：这里只导入了 flag_gems，还没有启用 FlagGems 算子。")


# In[4]:


assert sys.version_info >= (3, 10), (
    f"Swift 要求 Python >= 3.10，当前版本是 {sys.version}"
)

install_swift_command = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-cache-dir",
    "--no-deps",
    "-e",
    str(ROOT),
]

print("即将执行：")
print(" ".join(install_swift_command))

subprocess.check_call(install_swift_command)

# 保证当前 Notebook Kernel 不重启也能立即看到源码
src_path = str(ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("\nSwift 源码安装完成。")


# In[5]:


from packaging.version import Version

RUNTIME_REQUIREMENTS = {
    # 导入模块名: pip 安装名
    "numpy": "numpy",
    "h5py": "h5py>=3.13.0",
    "omegaconf": "omegaconf",
    "hydra": "hydra-core",
    "einops": "einops",
    "tqdm": "tqdm",
    "requests": "requests",
}

INSTALL_NOW = []

for module_name, pip_spec in RUNTIME_REQUIREMENTS.items():
    if importlib.util.find_spec(module_name) is None:
        INSTALL_NOW.append(pip_spec)
        print(f"[缺失] {module_name} -> 准备安装 {pip_spec}")
    else:
        print(f"[已有] {module_name}")

# Swift 源码要求 h5py >= 3.13.0
if importlib.util.find_spec("h5py") is not None:
    try:
        current_h5py = Version(metadata.version("h5py"))
        if current_h5py < Version("3.13.0"):
            if "h5py>=3.13.0" not in INSTALL_NOW:
                INSTALL_NOW.append("h5py>=3.13.0")
            print(
                f"[版本偏低] h5py={current_h5py}，"
                "源码要求至少为 3.13.0"
            )
    except Exception as exc:
        print("无法判断 h5py 版本：", exc)

# 去重，同时保持顺序
INSTALL_NOW = list(dict.fromkeys(INSTALL_NOW))
INSTALLED_THIS_RUN = INSTALL_NOW.copy()

print("\n本次需要安装：", INSTALL_NOW if INSTALL_NOW else "无")

if INSTALL_NOW:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        *INSTALL_NOW,
    ]

    print("\n执行：")
    print(" ".join(command))
    subprocess.check_call(command)
else:
    print("不需要安装额外运行包。")

print("\n重新检查导入：")

for module_name in RUNTIME_REQUIREMENTS:
    module = importlib.import_module(module_name)
    print(
        f"[成功] {module_name}:",
        getattr(module, "__version__", "未提供版本号")
    )


# In[6]:


MODEL_OPTIONS = {
    "swift": {
        "display_name": "Swift",
        "base": "weights/swift/020000",
        "checkpoint": "checkpoint-020000.pt",
    },
    "swift-b": {
        "display_name": "Swift-B",
        "base": "weights/swift/015000",
        "checkpoint": "checkpoint-015000.pt",
    },
}

# 默认验证 README 中的主 Swift 模型
MODEL_CHOICE = "swift"

assert MODEL_CHOICE in MODEL_OPTIONS

MODEL_INFO = MODEL_OPTIONS[MODEL_CHOICE]
MODEL_BASE = MODEL_INFO["base"]
CHECKPOINT_NAME = MODEL_INFO["checkpoint"]

CONFIG_PATH = ROOT / MODEL_BASE / ".hydra" / "config.yaml"
CHECKPOINT_PATH = ROOT / MODEL_BASE / "checkpoints" / CHECKPOINT_NAME
DATA_ROOT = ROOT / "sample-data"

free_gb = shutil.disk_usage(ROOT).free / 1024**3

print("选择的模型：", MODEL_INFO["display_name"])
print("配置文件目标位置：", CONFIG_PATH)
print("Checkpoint 目标位置：", CHECKPOINT_PATH)
print("样例数据目标位置：", DATA_ROOT)
print(f"当前磁盘剩余：{free_gb:.2f} GB")

if MODEL_CHOICE == "swift" and free_gb < 6:
    print("\n警告：磁盘空间可能不足。")
    print('可以把 MODEL_CHOICE 改为 "swift-b"，然后重新执行本单元。')


# In[7]:


from urllib.parse import quote
import requests
from tqdm.auto import tqdm

HF_REPO = "stockeh/swift-era5-1.4"

REQUIRED_REMOTE_FILES = [
    f"{MODEL_BASE}/.hydra/config.yaml",
    f"{MODEL_BASE}/checkpoints/{CHECKPOINT_NAME}",
    "sample-data/normalize_mean.npz",
    "sample-data/normalize_std.npz",
    "sample-data/normalize_diff_std_6.npz",
    "sample-data/test/2020_0937.h5",
    "sample-data/test/2020_0938.h5",
]

def download_hf_file(relative_path: str) -> Path:
    """
    从 Hugging Face 直接下载到源码目录。
    支持断点续传。
    下载中使用 .part，完成后自动改为正式文件名。
    """
    destination = ROOT / relative_path

    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"[跳过] {relative_path} "
            f"({destination.stat().st_size / 1024**2:.2f} MB)"
        )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(str(destination) + ".part")
    resume_size = temporary.stat().st_size if temporary.exists() else 0

    encoded_path = quote(relative_path, safe="/")
    url = (
        f"https://huggingface.co/{HF_REPO}/resolve/main/"
        f"{encoded_path}?download=true"
    )

    def make_request(offset: int):
        headers = {}
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"

        return requests.get(
            url,
            headers=headers,
            stream=True,
            allow_redirects=True,
            timeout=(30, 600),
        )

    response = make_request(resume_size)

    # 服务端不接受 Range 时，从头下载
    if resume_size > 0 and response.status_code != 206:
        response.close()
        temporary.unlink(missing_ok=True)
        resume_size = 0
        response = make_request(0)

    response.raise_for_status()

    remaining = int(response.headers.get("content-length") or 0)
    total = resume_size + remaining if remaining > 0 else None
    mode = "ab" if resume_size > 0 else "wb"

    with response:
        with temporary.open(mode) as output:
            with tqdm(
                total=total,
                initial=resume_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            ) as progress:
                for chunk in response.iter_content(
                    chunk_size=8 * 1024 * 1024
                ):
                    if not chunk:
                        continue

                    output.write(chunk)
                    progress.update(len(chunk))

    temporary.replace(destination)

    print(
        f"[完成] {relative_path} "
        f"({destination.stat().st_size / 1024**2:.2f} MB)"
    )

    return destination

print("开始检查或下载必需文件：\n")

for relative_path in REQUIRED_REMOTE_FILES:
    download_hf_file(relative_path)

print("\n全部必需文件处理完成。")


# In[8]:


print("文件完整性初步检查：\n")

for relative_path in REQUIRED_REMOTE_FILES:
    path = ROOT / relative_path

    assert path.exists(), f"文件不存在：{path}"
    assert path.stat().st_size > 1000, f"文件过小，可能下载失败：{path}"

    print(
        f"[存在] {relative_path:<75} "
        f"{path.stat().st_size / 1024**2:>10.2f} MB"
    )

assert CHECKPOINT_PATH.stat().st_size > 1024**3, (
    "Checkpoint 小于 1 GB，可能没有下载完整。"
)

print("\n文件初步检查通过。")


# In[9]:


import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate

np.random.seed(1118)
torch.manual_seed(1118)

cfg = OmegaConf.load(CONFIG_PATH)

print("配置中原始数据路径：")
print(cfg.data.dataset.root)

# 只修改内存中的 cfg，不改写 config.yaml
OmegaConf.update(
    cfg,
    "data.dataset.root",
    str(DATA_ROOT),
    merge=False,
)

# 当前只有两个连续样本，只进行 6 小时一步预测
OmegaConf.update(
    cfg,
    "data.dataset.intervals",
    [6],
    merge=False,
)

# Notebook 单进程读取，避免多进程 DataLoader 干扰
OmegaConf.update(
    cfg,
    "data.data_workers",
    0,
    merge=False,
)

print("\n内存中修改后的数据路径：")
print(cfg.data.dataset.root)

print("测试时间间隔：", list(cfg.data.dataset.intervals))
print("模型类：", cfg.model._target_)
print("预处理/预条件类：", cfg.precond._target_)
print("模型维度 dim：", cfg.model.dim)
print("模型深度 depth：", cfg.model.depth)
print("注意：磁盘上的 config.yaml 没有被修改。")


# In[10]:


dataset = instantiate(
    cfg.data.dataset,
    split="test",
    _convert_="object",
)

print("数据根目录：", dataset.root)
print("找到的 H5 文件：")

for file_path in dataset.files:
    print(" -", Path(file_path).name)

print("\n可用数据集长度：", len(dataset))
print("天气变量数量：", len(dataset.variables))
print("强迫变量数量：", len(dataset.forcings))
print("目标通道数量：", dataset.n_target_channels)
print("条件通道数量：", dataset.n_condition_channels)
print("空间大小：", dataset.img_resolution)
print("Residual 模式：", dataset.residual)

assert len(dataset) >= 1, (
    "数据集长度为 0。\n"
    "请确认：\n"
    "1. 两个 H5 文件都下载成功；\n"
    "2. cfg.data.dataset.intervals 已设置为 [6]。"
)

# 明确指定：
# idx=0，offset=1，delta=6 小时
(x, target_residual), (sample_index, auxiliary) = dataset[(0, 1, 6)]

print("\n单个输入张量形状：", tuple(x.shape))
print("单个目标张量形状：", tuple(target_residual.shape))
print("样本编号：", sample_index)
print("辅助时间条件：", auxiliary.item())

assert torch.isfinite(x).all(), "输入中出现 NaN 或 Inf。"
assert torch.isfinite(target_residual).all(), "目标中出现 NaN 或 Inf。"

assert x.shape[-2:] == torch.Size([128, 256]), (
    f"空间尺寸异常：{tuple(x.shape[-2:])}"
)

print("\n样例数据加载成功，输入和目标均为有限数值。")


# In[11]:


# FlagGems 官方示例使用 flag_gems.device 作为设备
raw_device = getattr(flag_gems, "device", None)

assert raw_device is not None, "flag_gems 没有提供 device。"

DEVICE = (
    raw_device
    if isinstance(raw_device, torch.device)
    else torch.device(raw_device)
)

print("即将使用的设备：", DEVICE)

# 尽可能匹配原始配置
if hasattr(torch, "set_float32_matmul_precision"):
    precision = str(
        getattr(cfg.system.torch, "set_float32_matmul_precision", "high")
    )
    torch.set_float32_matmul_precision(precision)
    print("float32 matmul precision：", precision)

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = bool(
        getattr(cfg.system.torch, "allow_tf32", True)
    )
    torch.backends.cudnn.benchmark = bool(
        getattr(cfg.system.torch, "benchmark", True)
    )

print("\n开始在 CPU 上构造模型……")

net = instantiate(
    cfg.precond,
    model_config=cfg.model,
    img_resolution=dataset.img_resolution,
    img_channels=dataset.n_target_channels,
    condition_channels=dataset.n_condition_channels,
    sigma_max=float("inf"),
    _recursive_=False,
    _convert_="object",
)

parameter_count = sum(parameter.numel() for parameter in net.parameters())

print(f"模型参数量：{parameter_count:,}")
print(f"约为：{parameter_count / 1e6:.2f} M 参数")

def load_checkpoint_compatibly(path: Path):
    """
    优先使用 weights_only 和 mmap。
    对较旧或厂商定制 PyTorch 提供兼容回退。
    """
    attempts = [
        {"weights_only": True, "mmap": True},
        {"weights_only": True},
        {},
    ]

    last_error = None

    for kwargs in attempts:
        try:
            print("尝试 torch.load 参数：", kwargs)
            return torch.load(
                path,
                map_location="cpu",
                **kwargs,
            )
        except (TypeError, RuntimeError) as exc:
            last_error = exc
            print("本次加载方式不可用：", repr(exc))

    raise RuntimeError(
        f"所有 checkpoint 加载方式均失败。最后错误：{last_error!r}"
    )

print("\n开始读取 checkpoint：")
print(CHECKPOINT_PATH)

state = load_checkpoint_compatibly(CHECKPOINT_PATH)

assert isinstance(state, dict), "Checkpoint 顶层不是字典。"
assert "ema" in state, "Checkpoint 中没有找到 ema 权重。"

print("Checkpoint 顶层键：", list(state.keys()))

load_result = net.load_state_dict(state["ema"], strict=True)

print("权重加载结果：", load_result)

del state
gc.collect()

print("\n将模型移动到加速设备……")

net = net.to(DEVICE)
net.eval()

print("模型当前设备：", next(net.parameters()).device)
print("模型 checkpoint 加载完成。")


# In[12]:


from swift.generating.factory import sampler_factory

# x 已经包含：
# [天气状态变量, 外部强迫变量]
condition = x.unsqueeze(0).to(DEVICE)

print("模型条件输入形状：", tuple(condition.shape))
print("模型要求的条件通道：", dataset.n_condition_channels)

assert condition.shape[1] == dataset.n_condition_channels, (
    "输入条件通道数与模型配置不一致。"
)

solver_kwargs = {
    "num_steps": 1,
    "sigma_min": 0.02,
    "sigma_max": 200.0,
    "auxiliary": 0.6,  # 6 小时 / 10
}

sampler = sampler_factory(
    "scm",
    net,
    **solver_kwargs,
)

def synchronize_device():
    backend = getattr(torch, DEVICE.type, None)
    synchronize = getattr(backend, "synchronize", None)

    if callable(synchronize):
        synchronize()

def reset_peak_memory():
    backend = getattr(torch, DEVICE.type, None)
    reset_fn = getattr(backend, "reset_peak_memory_stats", None)

    if callable(reset_fn):
        try:
            reset_fn()
        except Exception:
            pass

def get_peak_memory_gb():
    backend = getattr(torch, DEVICE.type, None)
    memory_fn = getattr(backend, "max_memory_allocated", None)

    if callable(memory_fn):
        try:
            return memory_fn() / 1024**3
        except Exception:
            return None

    return None

def run_one_step(seed: int = 0):
    """
    用固定随机种子完成一次 SCM 一步推理。
    返回 CPU float32 输出、运行时间和峰值显存。
    """
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    reset_peak_memory()
    synchronize_device()

    start = time.perf_counter()

    with torch.inference_mode():
        output = sampler(
            condition.clone(),
            generator=generator,
        )

    synchronize_device()

    elapsed = time.perf_counter() - start
    peak_memory = get_peak_memory_gb()

    output_cpu = output.detach().float().cpu()

    return output_cpu, elapsed, peak_memory

print("一步推理函数构造完成。")
print("此时 FlagGems 尚未启用。")


# In[13]:


print("开始原生 PyTorch 基线推理……")

baseline_output, baseline_time, baseline_peak_memory = run_one_step(seed=0)

print("\n原生 PyTorch 推理完成。")
print("输出形状：", tuple(baseline_output.shape))
print("运行时间：", f"{baseline_time:.6f} 秒")
print("输出最小值：", baseline_output.min().item())
print("输出最大值：", baseline_output.max().item())
print("输出平均值：", baseline_output.mean().item())
print("输出标准差：", baseline_output.std().item())
print("全部为有限数值：", bool(torch.isfinite(baseline_output).all()))

if baseline_peak_memory is not None:
    print("峰值设备内存：", f"{baseline_peak_memory:.3f} GB")

assert torch.isfinite(baseline_output).all(), (
    "原生 PyTorch 输出存在 NaN 或 Inf，不能继续测试 FlagGems。"
)


# In[14]:


variable_count = len(dataset.variables)

# 输入中的天气状态部分，不包括 forcing
initial_state_std = x[:variable_count].unsqueeze(0).cpu()

initial_state_physical = dataset.unstandardize_x(
    initial_state_std,
    delta=6,
)

true_residual_physical = dataset.unstandardize_t(
    target_residual.unsqueeze(0).cpu(),
    delta=6,
)

baseline_residual_physical = dataset.unstandardize_t(
    baseline_output,
    delta=6,
)

true_next_state = initial_state_physical + true_residual_physical
baseline_next_state = (
    initial_state_physical + baseline_residual_physical
)

t2m_index = dataset.variables.index("2m_temperature")

baseline_t2m_rmse = torch.sqrt(
    torch.mean(
        (
            baseline_next_state[:, t2m_index]
            - true_next_state[:, t2m_index]
        ) ** 2
    )
).item()

print("原生 PyTorch 2m_temperature 单样本 RMSE：")
print(baseline_t2m_rmse)


# In[15]:


from pathlib import Path
import flag_gems

GEMS_LOG_PATH = ROOT / "gems_debug.log"

# 删除上一次失败运行留下的旧日志
if GEMS_LOG_PATH.exists():
    GEMS_LOG_PATH.unlink()
    print("已删除旧日志：", GEMS_LOG_PATH)

# 本次实际禁用并回退到 PyTorch 的算子
UNUSED_OPS = [
    "batch_norm",
    "batch_norm_backward",

    # 当前 GPU 的共享内存上限为 65536 Bytes，
    # FlagGems mm 内核要求 131072 Bytes，因此禁用
    "mm",
]

print("准备启用 FlagGems")
print("回退到 PyTorch 的算子：", UNUSED_OPS)

flag_gems.enable(
    unused=UNUSED_OPS,
    record=True,
    path=str(GEMS_LOG_PATH),
    once=True,
)

print("\nFlagGems 已启用")
print("日志路径：", GEMS_LOG_PATH)
print("mm 将使用 PyTorch 原生实现")


# In[16]:


import torch

device = torch.device("cuda")

a = torch.randn(
    256,
    256,
    device=device,
    dtype=torch.float16,
)

b = torch.randn(
    256,
    256,
    device=device,
    dtype=torch.float16,
)

try:
    with torch.inference_mode():
        c = torch.mm(a, b)

    torch.cuda.synchronize()

    print("torch.mm 执行成功")
    print("输出形状：", tuple(c.shape))
    print("输出设备：", c.device)
    print("输出是否有限：", bool(torch.isfinite(c).all()))

except Exception as exc:
    print("torch.mm 仍然失败")
    print("错误类型：", type(exc).__name__)
    print("错误内容：", repr(exc))
    raise


# In[17]:


import torch

assert torch.cuda.is_available(), "当前 CUDA 不可用"

device_index = torch.cuda.current_device()
properties = torch.cuda.get_device_properties(device_index)

print("CUDA 设备编号：", device_index)
print("GPU 型号：", properties.name)
print("CUDA Compute Capability：", properties.major, properties.minor)
print(
    "总显存：",
    f"{properties.total_memory / 1024**3:.2f} GB",
)

shared_memory = getattr(
    properties,
    "shared_memory_per_block",
    None,
)

shared_memory_optin = getattr(
    properties,
    "shared_memory_per_block_optin",
    None,
)

print(
    "每个 block 默认共享内存：",
    shared_memory,
    "Bytes",
)

print(
    "每个 block opt-in 共享内存：",
    shared_memory_optin,
    "Bytes",
)

if shared_memory is not None:
    print(
        "每个 block 默认共享内存：",
        f"{shared_memory / 1024:.2f} KiB",
    )

if shared_memory_optin is not None:
    print(
        "每个 block opt-in 共享内存：",
        f"{shared_memory_optin / 1024:.2f} KiB",
    )


# In[18]:


print("开始 FlagGems 推理……")

gems_output, gems_time, gems_peak_memory = run_one_step(seed=0)

print("\nFlagGems 推理完成。")
print("输出形状：", tuple(gems_output.shape))
print("运行时间：", f"{gems_time:.6f} 秒")
print("输出最小值：", gems_output.min().item())
print("输出最大值：", gems_output.max().item())
print("输出平均值：", gems_output.mean().item())
print("输出标准差：", gems_output.std().item())
print("全部为有限数值：", bool(torch.isfinite(gems_output).all()))

if gems_peak_memory is not None:
    print("峰值设备内存：", f"{gems_peak_memory:.3f} GB")

assert tuple(gems_output.shape) == tuple(baseline_output.shape), (
    "FlagGems 与原生 PyTorch 输出形状不一致。"
)

assert torch.isfinite(gems_output).all(), (
    "FlagGems 输出出现 NaN 或 Inf。"
)

absolute_difference = torch.abs(
    gems_output - baseline_output
)

max_absolute_error = absolute_difference.max().item()
mean_absolute_error = absolute_difference.mean().item()

baseline_norm = torch.linalg.vector_norm(
    baseline_output.reshape(-1)
).clamp_min(1e-12)

relative_l2_error = (
    torch.linalg.vector_norm(
        (gems_output - baseline_output).reshape(-1)
    )
    / baseline_norm
).item()

preliminary_allclose = torch.allclose(
    gems_output,
    baseline_output,
    rtol=1e-2,
    atol=1e-2,
)

print("\n" + "=" * 70)
print("FlagGems 与原生 PyTorch 数值对比")
print("=" * 70)
print("最大绝对误差：", max_absolute_error)
print("平均绝对误差：", mean_absolute_error)
print("相对 L2 误差：", relative_l2_error)
print("初步 allclose(rtol=1e-2, atol=1e-2)：", preliminary_allclose)

gems_residual_physical = dataset.unstandardize_t(
    gems_output,
    delta=6,
)

gems_next_state = (
    initial_state_physical + gems_residual_physical
)

gems_t2m_rmse = torch.sqrt(
    torch.mean(
        (
            gems_next_state[:, t2m_index]
            - true_next_state[:, t2m_index]
        ) ** 2
    )
).item()

print("\n2m_temperature 单样本 RMSE")
print("原生 PyTorch：", baseline_t2m_rmse)
print("FlagGems：", gems_t2m_rmse)
print(
    "二者 RMSE 差值：",
    abs(gems_t2m_rmse - baseline_t2m_rmse),
)

FUNCTIONAL_PASS = bool(
    torch.isfinite(gems_output).all()
    and tuple(gems_output.shape) == tuple(baseline_output.shape)
)

PRELIMINARY_NUMERICAL_PASS = bool(
    FUNCTIONAL_PASS and preliminary_allclose
)

print("\n功能运行是否通过：", FUNCTIONAL_PASS)
print("初步数值一致性是否通过：", PRELIMINARY_NUMERICAL_PASS)


# In[19]:


import logging

# 强制刷新 Python 日志缓冲区
for logger_object in [
    logging.getLogger(),
    *[
        obj
        for obj in logging.Logger.manager.loggerDict.values()
        if isinstance(obj, logging.Logger)
    ],
]:
    for handler in logger_object.handlers:
        try:
            handler.flush()
        except Exception:
            pass

print("日志路径：", GEMS_LOG_PATH)
print("日志是否存在：", GEMS_LOG_PATH.exists())

assert GEMS_LOG_PATH.exists(), (
    "没有生成 gems_debug.log。\n"
    "请确认：\n"
    "1. FlagGems enable 单元在 FlagGems 推理之前执行；\n"
    "2. FlagGems 推理单元确实完成；\n"
    "3. path 指向当前有写权限的源码目录。"
)

log_size = GEMS_LOG_PATH.stat().st_size

print("日志大小：", log_size, "Bytes")

assert log_size > 0, (
    "gems_debug.log 存在但为空，说明没有记录到有效算子。"
)

log_text = GEMS_LOG_PATH.read_text(
    encoding="utf-8",
    errors="replace",
)

log_lines = log_text.splitlines()

print("日志行数：", len(log_lines))

print("\n日志前 80 行：")
print("-" * 70)
print("\n".join(log_lines[:80]))

if len(log_lines) > 80:
    print("\n日志最后 20 行：")
    print("-" * 70)
    print("\n".join(log_lines[-20:]))

print("\nFlagGems 日志检查通过。")


# In[20]:


def version_or_unknown(module_name, distribution_name=None):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if version is not None:
            return str(version)
    except Exception:
        pass

    try:
        return metadata.version(distribution_name or module_name)
    except Exception:
        return "未知"

device_description = str(DEVICE)

if DEVICE.type == "cuda" and torch.cuda.is_available():
    device_description = torch.cuda.get_device_name(DEVICE)

installed_text = (
    ", ".join(INSTALLED_THIS_RUN)
    if INSTALLED_THIS_RUN
    else "未额外安装运行依赖"
)

report_summary = f"""
# Swift / FlagGems 模型适配测试摘要

## 1. 模型信息

- 模型仓库：stockeh/swift-era5-1.4
- 验证变体：{MODEL_INFO["display_name"]}
- 配置目录：{MODEL_BASE}
- Checkpoint：{CHECKPOINT_NAME}
- 测试任务：ERA5 1.40625° 天气预报
- 测试规模：1 个样本、1 个随机成员、1 个 6 小时预测步
- 输入空间大小：{dataset.img_resolution}
- 输入条件形状：{tuple(condition.shape)}
- 模型输出形状：{tuple(gems_output.shape)}

## 2. 软件环境

- Python：{platform.python_version()}
- PyTorch：{torch.__version__}
- Triton：{version_or_unknown("triton")}
- FlagGems：{version_or_unknown("flag_gems", "flag-gems")}
- h5py：{version_or_unknown("h5py")}
- Hydra：{version_or_unknown("hydra", "hydra-core")}
- OmegaConf：{version_or_unknown("omegaconf")}
- 设备：{device_description}

## 3. 安装记录

- Swift 安装命令：
  {sys.executable} -m pip install --no-cache-dir --no-deps -e {ROOT}
- 本次补充安装的包：
  {installed_text}
- 未重新安装 torch、triton、flag_gems。

## 4. FlagGems 设置

- unused：
  batch_norm
  batch_norm_backward
- record：True
- once：True
- 日志：{GEMS_LOG_PATH}
- 日志大小：{log_size} Bytes
- 日志行数：{len(log_lines)}

## 5. 原生 PyTorch 结果

- 成功运行：True
- 输出全部为有限值：{bool(torch.isfinite(baseline_output).all())}
- 首次运行时间：{baseline_time:.6f} 秒
- 峰值设备内存：{baseline_peak_memory}
- 2m_temperature 单样本 RMSE：{baseline_t2m_rmse}

## 6. FlagGems 结果

- 成功运行：{FUNCTIONAL_PASS}
- 输出全部为有限值：{bool(torch.isfinite(gems_output).all())}
- 首次运行时间：{gems_time:.6f} 秒
- 峰值设备内存：{gems_peak_memory}
- 2m_temperature 单样本 RMSE：{gems_t2m_rmse}

## 7. 数值一致性

- 最大绝对误差：{max_absolute_error}
- 平均绝对误差：{mean_absolute_error}
- 相对 L2 误差：{relative_l2_error}
- allclose，rtol=1e-2，atol=1e-2：{preliminary_allclose}

## 8. 初步结论

- 功能适配：{"通过" if FUNCTIONAL_PASS else "未通过"}
- 初步数值一致性：{"通过" if PRELIMINARY_NUMERICAL_PASS else "需要进一步分析"}
- 本次属于单样本最小功能验证，不代表完整数据集科学精度或正式性能结果。
"""

print(report_summary)

