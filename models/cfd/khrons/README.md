---
# 用户自定义标签
tags:
  - kernel-based learning
  - multi-fidelity prediction
  - aerodynamic field prediction
  - resource-efficient surrogate
---

<h1 align="center">KHRONOS</h1>

## 模型介绍

KHRONOS 是一种基于核方法的资源高效神经代理模型，用于多保真度气动场预测。该方法结合核方法与神经网络，能够在不同保真度数据之间实现高效的知识迁移，显著降低训练所需的计算资源，同时保持较高的预测精度，适用于大规模气动场预测任务。

**论文**: A Kernel-based Resource-efficient Neural Surrogate for Multi-fidelity Prediction of Aerodynamic Field (Sarker et al., arXiv 2025)

**模型参数**: 22,781  |  **Test Loss**: 0.739

## 模型描述

- `model/model.py` — 模型定义代码
- `weight/best_model.pt` — 最佳模型权重
- `weight/final_model.pt` — 最终模型权重
- `conf/results.json` — 实验结果

## 使用说明

### 环境安装

```bash
pip install torch numpy
```

### 下载模型

```bash
git clone https://www.modelscope.cn/OneScience/KHRONOS.git
```

### 训练

```bash
cd KHRONOS
python train.py
```

### 推理

```python
import torch
from model.model import KHRONOS

model = KHRONOS()
model.load_state_dict(torch.load('weight/best_model.pt'))
model.eval()
```

## 自定义标签

`kernel-method` `multi-fidelity-prediction` `aerodynamic-surrogate` `resource-efficient-neural-network` `tensor-decomposition`

## OneScience 官方信息

本项目由 **OneScience** 团队维护，致力于推动科学计算与人工智能的交叉研究。

- ModelScope: [https://www.modelscope.cn/organization/OneScience](https://www.modelscope.cn/organization/OneScience)

## 许可证

本项目代码遵循 MIT 许可证。
