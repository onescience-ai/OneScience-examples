"""配置加载工具。"""

from __future__ import annotations

import os
import yaml
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    """从 yaml 加载配置。路径缺失时报错。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(p: str) -> str:
    """将相对路径基于当前工作目录解析为绝对路径（供 CLI 使用）。"""
    return os.path.abspath(p)
