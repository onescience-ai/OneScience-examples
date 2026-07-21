import importlib
import sys

# (pip安装名, 导入名)
packages = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("gradio", "gradio"),
    ("herbie-data", "herbie"),  # pip 名叫 herbie-data，但 import 叫 herbie
    ("cfgrib", "cfgrib"),
    ("xarray", "xarray"),
    ("eccodes", "eccodes"),
    ("cartopy", "cartopy"),
    ("scipy", "scipy"),
]

print("=" * 50)
print("📦 DCU 环境依赖检查报告")
print("=" * 50)

missing = []
for pip_name, import_name in packages:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "未知版本")
        print(f"✅ {pip_name:15} -> 已安装 (版本: {version})")
    except ImportError as e:
        print(f"❌ {pip_name:15} -> 未安装或导入失败")
        missing.append(pip_name)

print("=" * 50)
if missing:
    print(f"⚠️  缺少以下 {len(missing)} 个包: {', '.join(missing)}")
    print("请用 pip install 安装它们")
else:
    print("🎉 所有依赖包都已就绪！可以运行 python app.py")
