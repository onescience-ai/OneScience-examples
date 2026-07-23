# -*- coding: utf-8 -*-
"""
diagnose.py —— 只在 run.sh 跑不通时才需要执行的排查脚本。
    python diagnose.py          # 结果同时写入 diagnose.log

它会:
 1. 打印环境版本;
 2. 逐个 import 可疑的库(tensorflow / xformers / onnxruntime / accelerate /
    transformers / flag_gems / diffusers),各自打印**完整 traceback**,
    从而定位到底是哪一层把 libtensorflow_cc.so.2 拖进来的;
 3. 分别在「屏蔽 tensorflow」前后测试 `from diffusers import VQModel`;
 4. 直接用 torch.load 打开权重文件,打印全部参数名与形状(用于核对结构)。

把 diagnose.log 整个发回即可定位问题。
"""

import os
import sys
import types
import importlib
import importlib.machinery
import traceback

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnose.log")


class _Tee:
    def __init__(self, *s): self.s = s

    def write(self, d):
        for x in self.s:
            x.write(d); x.flush()

    def flush(self):
        for x in self.s:
            x.flush()


_f = open(LOG, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _f)
sys.stderr = _Tee(sys.__stderr__, _f)


def sec(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def try_import(name):
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "")
        print(f"  [OK]   import {name}  {v}")
        return True
    except Exception:
        print(f"  [FAIL] import {name}")
        print(traceback.format_exc())
        return False


sec("1. 基本环境")
print(f"python     : {sys.version}")
print(f"executable : {sys.executable}")
for k in ("USE_TF", "USE_TORCH", "USE_JAX", "LD_LIBRARY_PATH", "LD_PRELOAD"):
    print(f"env {k:16s}= {os.environ.get(k, '(unset)')}")

sec("2. 逐个导入可疑的库(未屏蔽 tensorflow)")
for mod in ("torch", "numpy", "sklearn", "safetensors", "huggingface_hub",
            "tensorflow", "xformers", "onnxruntime", "accelerate",
            "transformers", "flag_gems", "diffusers"):
    try_import(mod)

sec("3. from diffusers import VQModel(未屏蔽 tensorflow)")
try:
    from diffusers import VQModel  # noqa: F401
    print("  [OK] 成功")
except Exception:
    print("  [FAIL]")
    print(traceback.format_exc())

sec("4. 屏蔽 tensorflow 后重试(需重开进程才最干净,此处尽力而为)")
if "tensorflow" not in sys.modules:
    stub = types.ModuleType("tensorflow")
    stub.__version__ = "0.0.0-stub"
    stub.__path__ = []
    stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
    sys.modules["tensorflow"] = stub
    print("  已安装 tensorflow 占位模块")
else:
    print("  tensorflow 已在 sys.modules 中(可能已被导入过)")

for m in [k for k in list(sys.modules) if k.startswith("diffusers")]:
    sys.modules.pop(m, None)
try:
    from diffusers import VQModel  # noqa: F401,F811
    print("  [OK] 屏蔽后成功")
except Exception:
    print("  [FAIL] 屏蔽后仍失败")
    print(traceback.format_exc())

sec("5. 权重文件参数清单(核对结构用)")
here = os.path.dirname(os.path.abspath(__file__))
wpath = None
for fn in ("diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"):
    p = os.path.join(here, fn)
    if os.path.exists(p):
        wpath = p
        break
if wpath is None:
    print("  未找到权重文件。")
else:
    print(f"  文件: {wpath}  ({os.path.getsize(wpath):,} bytes)")
    try:
        import torch
        if wpath.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(wpath)
        else:
            sd = torch.load(wpath, map_location="cpu")
        total = 0
        print(f"  张量数: {len(sd)}")
        for k, v in sd.items():
            total += v.numel()
            print(f"    {k:<72} {tuple(v.shape)}")
        print(f"  参数总量: {total:,}")
    except Exception:
        print(traceback.format_exc())

print(f"\n[done] 诊断结果已写入 {LOG}")
