---
# 用户自定义标签
tags:
  - physics-informed learning
  - geometric operator learning
  - complex PDE solving
  - fluid dynamics
---

<h1 align="center">PGOT</h1>

## 模型介绍

PGOT（Physics-Geometry Operator Transformer）是一种物理-几何算子Transformer，用于求解复杂偏微分方程。该方法将物理信息与几何信息深度融合到Transformer架构中，通过算子学习的方式实现对复杂PDEs的高效求解，在流体力学等领域具有广泛应用前景。

**论文**: PGOT: A Physics-Geometry Operator Transformer for Complex PDEs (Zhang et al., arXiv 2026)

**模型参数**: 101,636  |  **Validation Loss**: 1.278（训练2个epoch）

## 模型描述

- `model/model.py` — 模型定义代码
- `weight/best_model.pt` — 最佳模型权重
- `conf/results.json` — 实验结果

## 使用说明

### 环境安装

```bash
pip install torch numpy
```

### 下载模型

```bash
git clone https://www.modelscope.cn/OneScience/PGOT.git
```

### 训练

```bash
cd PGOT
python train.py
```

### 推理

```python
import torch
from model.model import PGOT

model = PGOT()
model.load_state_dict(torch.load('weight/best_model.pt'))
model.eval()
```

## 自定义标签

`physics-geometry-operator` `operator-transformer` `pde-solver` `specgeo-attention` `physics-informed-learning`

## OneScience 官方信息

本项目由 **OneScience** 团队维护，致力于推动科学计算与人工智能的交叉研究。

- ModelScope: [https://www.modelscope.cn/organization/OneScience](https://www.modelscope.cn/organization/OneScience)

## 许可证

本项目代码遵循 MIT 许可证。
