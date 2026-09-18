<p align="center">
  <strong>
    <span style="font-size: 30px;">KNO</span>
  </strong>
</p>

# 模型介绍

KNO（Koopman Neural Operator）是一种基于 Koopman 算子的神经算子，用于学习非线性动力系统的演化规律。本模型面向二维规则网格上的 Navier-Stokes 时序预测，默认使用前 10 个时间步预测后续 10 个时间步。

论文：Koopman Neural Operator as a Mesh-free Solver of Non-linear Partial Differential Equations  
https://doi.org/10.1016/j.jcp.2024.113194

# 模型描述

KNO 使用编码器将历史物理场映射到 Koopman 隐空间，在 Fourier 频域内学习近似线性的演化算子，再通过解码器还原未来物理场。模型支持线性或非线性隐状态推进，并通过自回归方式完成多步流场预测。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 流场时序预测 | 根据历史涡量场预测未来 Navier-Stokes 状态。 |
| Koopman 算子研究 | 研究非线性动力系统在隐空间中的近似线性演化。 |
| CFD 快速代理 | 学习规则网格上历史物理场到未来物理场的映射。 |
| 模型流程验证 | 使用随包权重或小规模配置检查训练和推理流程。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 进行训练和推理。
- CPU 可用于小配置连通性验证，完整训练速度较慢。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

### 下载模型包

```bash
modelscope download --model OneScience/KNO --local_dir ./KNO
cd KNO
```

### 安装运行环境

**DCU 环境**

```bash
# 请首先激活 DTK 及 CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
# 请首先激活 CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

模型使用 Navier-Stokes 标准数据集 `NavierStokes_V1e-5_N1200_T20.mat`，数据变量 `u` 的形状为 `[1200, 64, 64, 20]`。可通过以下命令下载数据，并确认 `conf/config.yaml` 中的数据路径配置正确：

```bash
modelscope download --dataset OneScience/cfd_benchmark data/ns/NavierStokes_V1e-5_N1200_T20.mat --local_dir ./data
```

### 训练

```bash
python scripts/train.py
```

默认权重保存至 `weight/kno_navier_stokes.pt`。

### 训练权重

本仓库在`weight/`文件夹内提供基于Navier-Stokes 标准数据集训练的权重。

### 推理、评估和可视化

模型包提供用于流程验证的权重，可在准备数据后直接运行：

```bash
python scripts/inference.py
```

脚本默认读取 `weight/kno_navier_stokes.pt`，预测张量和可视化结果保存在 `result/` 目录。训练和推理参数均可在 `conf/config.yaml` 中修改。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Xiong, W. et al. Koopman Neural Operator as a Mesh-free Solver of Non-linear Partial Differential Equations. Journal of Computational Physics, 2024.
- Xiong, W. et al. KoopmanLab: Machine Learning for Solving Complex Physics Equations. APL Machine Learning, 2023.
- 模型实现源自 GPL-3.0 许可的 KoopmanLab，本模型包沿用 GPL-3.0 许可证并保留来源说明。
