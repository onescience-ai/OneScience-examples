"""
MatRIS: Material Representation Learning with Interatomic Structure

A deep learning model for predicting material properties including energy, 
forces, stress, and magnetic moments.
"""

import sys

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)  # read from pyproject.toml
except PackageNotFoundError:
    __version__ = "unknown"

# Import main model
from model.matris import MatRIS

# Import graph components
from onescience.datapipes.materials.matris import GraphConverter, RadiusGraph

# 把本仓库 model 注册为 onescience.models.matris，使得 OneScience 的 MatRISCalculator、
# StructOptimizer 等工具类在加载预训练权重时，直接使用本仓库的 MatRIS.load 方法。
# 默认权重会下载到本仓库根目录下的 weight/ 中。
sys.modules["onescience.models.matris"] = sys.modules[__name__]

__all__ = [
    "MatRIS",
    "GraphConverter", 
    "RadiusGraph",
    "__version__",
]
