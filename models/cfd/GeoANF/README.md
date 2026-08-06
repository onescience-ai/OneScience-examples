---
# 用户自定义标签
tags:
  - airfoil flow prediction
  - geometric attention
  - neural field
  - aerodynamics
---

<h1 align="center">GeoANF</h1>

## 模型介绍

GeoANF（Geometric Attention Neural Field）是一种用于翼型流场表示的几何注意力神经场模型。该方法通过几何注意力机制学习翼型周围的流场表示，能够高效地捕捉翼型几何与流场之间的复杂映射关系，在空气动力学应用中实现高精度的流场预测。

**论文**: Learning Airfoil Flow Field Representation via Geometric Attention Neural Field (Xiao et al., Applied Sciences 2024)

**模型参数**: 64,644  |  **Test Loss**: 0.471

## 模型描述

- `model/model.py` — 模型定义代码
- `scripts/train_standalone.py` — 独立训练脚本
- `scripts/train.py` — 训练脚本
- `scripts/run_job.sh` — 作业提交脚本
- `weight/best_model.pt` — 最佳模型权重
- `weight/final_model.pt` — 最终模型权重
- `conf/results.json` — 实验结果

## 使用说明

### 环境安装

```bash
pip install torch numpy matplotlib
```

### 下载模型

```bash
git clone https://www.modelscope.cn/OneScience/GeoANF.git
```

### 训练

```bash
cd GeoANF
python scripts/train_standalone.py
```

### 推理

```python
import torch
from model.model import GeoANF

model = GeoANF()
model.load_state_dict(torch.load('weight/best_model.pt'))
model.eval()
```

## 自定义标签

`geometric-attention` `neural-field` `airfoil-flow-prediction` `implicit-neural-representation` `aerodynamics-surrogate`

## OneScience 官方信息

本项目由 **OneScience** 团队维护，致力于推动科学计算与人工智能的交叉研究。

- ModelScope: [https://www.modelscope.cn/organization/OneScience](https://www.modelscope.cn/organization/OneScience)

## 许可证

本项目代码遵循 MIT 许可证。
