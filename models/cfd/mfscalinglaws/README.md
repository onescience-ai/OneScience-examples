---
# 用户自定义标签
tags:
  - multi-fidelity modeling
  - neural scaling laws
  - CFD surrogate modeling
  - model capacity analysis
---

<h1 align="center">MfScalingLaws</h1>

## 模型介绍

MfScalingLaws 致力于探索神经代理模型在计算流体力学中的多保真度缩放规律。该方法通过研究不同保真度数据下神经网络的性能变化规律，揭示了模型容量、数据保真度与预测精度之间的量化关系，为多保真度建模提供理论指导。

**论文**: Towards Multi-Fidelity Scaling Laws of Neural Surrogates in CFD (Setinek et al., AI for Science Workshop 2025)

**模型参数**: 135,305  |  **Test Loss**: 0.443

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
git clone https://www.modelscope.cn/OneScience/MfScalingLaws.git
```

### 训练

```bash
cd MfScalingLaws
python train.py
```

### 推理

```python
import torch
from model.model import MfScalingLaws

model = MfScalingLaws()
model.load_state_dict(torch.load('weight/best_model.pt'))
model.eval()
```

## 自定义标签

`scaling-laws` `multi-fidelity-cfd` `neural-surrogate` `fidelity-mixing` `data-efficiency-analysis`

## OneScience 官方信息

本项目由 **OneScience** 团队维护，致力于推动科学计算与人工智能的交叉研究。

- ModelScope: [https://www.modelscope.cn/organization/OneScience](https://www.modelscope.cn/organization/OneScience)

## 许可证

本项目代码遵循 MIT 许可证。
