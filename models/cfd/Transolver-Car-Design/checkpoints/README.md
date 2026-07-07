---
frameworks:
- "pytorch"
library_name: onescience
license: Apache-2.0
tags:
- OneScience
- Transolver
- Transolver++
- ShapeNetCar
- CFD
- AI for science
- car design
- fluid dynamics
- 空气动力学
- 汽车设计
---

<div>
  <p style="margin-bottom: 0; margin-top: 0;">
    <strong>学习如何使用基于 OneScience 训练的 Transolver++ 汽车空气动力学预测模型。</strong>
  </p>
  <p style="margin-bottom: 0;">
    项目框架：OneScience | 模型结构：Transolver++ / Physics-Attention Transformer
  </p>
  <h1 style="margin-top:0rem; margin-bottom: 0rem;">Transolver++ 汽车设计 CFD 预测模型</h1>
</div>

## 快速使用指南

- **环境要求**：Python 3.11+，PyTorch，OneScience，CUDA / DCU 可用环境
- **模型权重**：`ShapeNetCar/Transolver_plus.pth`
- **模型结构**：Transolver++，隐藏维度 `n_hidden=256`，层数 `n_layers=8`，注意力头数 `n_head=8`
- **任务类型**：三维汽车外流场 CFD 预测
- **预测目标**：三维速度场、表面压力与阻力系数
- **数据要求**：ShapeNetCar / mlcfd_data 数据集，包含汽车几何、表面网格、体场点和归一化统计信息
- **快速加载环境并推理**（使用 OneScience 平台，以DCU为例）：

  ```bash
  module load sghpcdas/25.6
  conda init bash
  source ~/.bashrc
  module load sghpc-mpi-gcc/26.3

  conda create -n onescience311 python=3.11
  conda activate onescience311

  cd onescience/examples/cfd/Transolver-Car-Design
  source ../../../env.sh

  mkdir -p checkpoints/ShapeNetCar
  cp /path/to/modelscope_snapshot/ShapeNetCar/Transolver_plus.pth checkpoints/ShapeNetCar/Transolver_plus.pth

  python inference.py
  ```

- **使用 ModelScope SDK 下载权重**：

  ```python
  from modelscope import snapshot_download

  model_dir = snapshot_download("your_namespace/your_model_name")
  print(model_dir)
  ```

# 1. 模型简介

本模型基于 **OneScience** 框架中的 Transolver++ 架构训练，用于汽车设计场景中的三维 CFD 流场预测。模型以汽车几何相关的空间特征、SDF、法向量和网格点信息为输入，预测车辆周围速度场与表面压力，并进一步用于计算汽车阻力系数。

Transolver 是一种面向通用几何 PDE 求解的 Transformer 模型，通过物理感知切片和 token 级注意力机制，将非结构化网格点自适应聚合为物理相关状态，从而降低注意力计算复杂度并提升几何泛化能力。Transolver++ 在此基础上引入局部自适应机制和切片重参数化，使模型能够更稳定地学习复杂汽车外流场中的局部物理状态。

# 2. 基于 OneScience 加载模型

模型权重由 OneScience 训练脚本保存，默认文件名为：

```text
ShapeNetCar/Transolver_plus.pth
```

在 Python 中可按如下方式加载：

```python
import torch
from onescience.models.transolver import Transolver3D_plus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transolver3D_plus(
    n_hidden=256,
    n_layers=8,
    space_dim=7,
    fun_dim=0,
    n_head=8,
    mlp_ratio=2,
    out_dim=4,
    slice_num=32,
    unified_pos=0,
).to(device)

checkpoint = torch.load("ShapeNetCar/Transolver_plus.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

完整推理流程请使用 OneScience 示例脚本 `inference.py`。该脚本会读取 `conf/transolver_car.yaml`，构建 `ShapeNetCarDatapipe`，加载 `checkpoints/ShapeNetCar/Transolver_plus.pth`，并输出压力、速度和阻力系数相关指标。若配置中启用 `save_vtk: True` 和 `visualize: True`，推理结果会保存到 `results/ShapeNetCar/Transolver_plus`，可使用 ParaView 查看 VTK 结果。

# 3. 参考

- Transolver: A Fast Transformer Solver for PDEs on General Geometries
- Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries
- Paper: https://arxiv.org/abs/2402.02366
- Transolver++ Paper: https://arxiv.org/pdf/2502.02414
- Project: https://github.com/thuml/Transolver/tree/main

## 4. 联系方式
如有任何问题或合作意向，请通过以下方式联系：

邮箱：songzhl@sugon.com

## OneScience 链接

### Gitee
- Doc：https://gitee.com/onescience-ai/onescience-doc
- OneScience：https://gitee.com/onescience-ai/onescience
- Skills：https://gitee.com/onescience-ai/oneskills

### GitHub
- Doc：https://github.com/onescience-ai/OneScience-doc
- OneScience：https://github.com/onescience-ai/OneScience
- Skills：https://github.com/onescience-ai/oneskills
