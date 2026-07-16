---
license: gpl-3.0
language:
- en
- zh
tags:
- OneScience
- KNO
- Koopman-operator
- neural-operator
- computational-fluid-dynamics
- Navier-Stokes
frameworks: PyTorch
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">KNO</span>
  </strong>
</p>

# 模型介绍

KNO（Koopman Neural Operator）使用编码器将历史物理场映射到 Koopman 隐空间，在 Fourier 频域内学习近似线性的 Koopman 演化算子，再通过解码器将隐状态还原为未来物理场。

本模型包面向二维规则网格 Navier-Stokes 时序预测。默认使用前 10 个时间步作为输入，通过自回归方式预测后续 10 个时间步。

论文：Koopman Neural Operator as a Mesh-free Solver of Non-linear Partial Differential Equations  
https://doi.org/10.1016/j.jcp.2024.113194

# 仓库说明

本仓库是 OneScience 整理的 KNO 最小可运行独立模型仓库，模型实现源自 KoopmanLab，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用统一 YAML 配置训练和推理二维 KNO
- 使用复数 Fourier Koopman 矩阵推进隐空间状态
- 支持线性残差推进或带 `tanh` 的非线性推进
- 支持可选 BatchNorm 隐状态归一化
- 对 Navier-Stokes 流场执行多步自回归预测
- 计算相对 L2 误差并保存预测张量和可视化结果
- 模型文件同时保留基础 KNO1D、KNO2D 和 Navier-Stokes 适配器

当前不支持能力：

- 当前源码没有 KNO3D 实现
- 不内置约 394 MiB 的 Navier-Stokes 原始数据文件
- 随包 checkpoint 仅用于流程验证，不代表论文报告的正式精度

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 流场时序预测 | 根据历史涡量场自回归预测未来 Navier-Stokes 状态 |
| Koopman 算子研究 | 研究非线性动力系统在隐空间中的近似线性演化 |
| CFD 快速代理 | 学习规则网格上的历史场到未来场映射 |
| 模型流程验证 | 使用随包 checkpoint 或单 epoch 配置检查推理训练流程 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 数据、模型、训练及推理配置 | 路径相对于模型包根目录解析 |
| `model/kno.py` | KNO1D、KNO2D 及 Navier-Stokes 适配器 | 独立 PyTorch 实现 |
| `scripts/common.py` | 配置、设备、指标和自回归公共函数 | 训练和推理共用 |
| `scripts/train.py` | 训练与验证脚本 | 保存最佳相对 L2 检查点 |
| `scripts/inference.py` | 推理、评估与可视化脚本 | 读取 `weight/*.pt` |
| `data/` | Navier-Stokes 数据目录 | 放置 benchmark `.mat` 文件 |
| `weight/kno_navier_stokes.pt` | 一轮训练流程验证 checkpoint | 可直接用于接口验证 |
| `result/` | 推理结果目录 | 首次推理时自动创建 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 完成正式训练。
- CPU 可用于缩小模型和数据后的流程验证，完整训练速度较慢。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

### 下载模型包

```bash
modelscope download --model OneScience/KNO --local_dir ./KNO
cd KNO
```

### 安装运行环境

已有 OneScience 环境时直接激活对应环境。新环境需安装 OneScience 及基础依赖：

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience torch scipy numpy pyyaml matplotlib tqdm
```

## 3. 快速开始

### 准备数据

从 Transolver `PDE-Solving-StandardBenchmark` 下载以下文件：

```text
NavierStokes_V1e-5_N1200_T20.mat
```

将文件放入 `data/`：

```text
data/
  NavierStokes_V1e-5_N1200_T20.mat
```

数据下载与说明：

- https://github.com/thuml/Transolver/tree/main/PDE-Solving-StandardBenchmark
- https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-

同时，OneScience社区提供可供训练的数据，用户可通过下述命令下载，并确认'conf/config.yaml'中数据路径设置正确；

```bash
modelscope download --dataset OneScience/cfd_benchmark  data/ns/NavierStokes_V1e-5_N1200_T20.mat  --local_dir ./data
```

### 训练

```bash
python scripts/train.py
```

默认检查点保存至 `weight/kno_navier_stokes.pt`。训练脚本完全由 `conf/config.yaml` 驱动，不接收命令行配置参数。


### 推理、评估和可视化

准备数据后可直接使用随包 checkpoint：

```bash
python scripts/inference.py
```

脚本默认读取 `weight/kno_navier_stokes.pt`，并在 `result/` 下生成：

```text
result/
  prediction_sample.pt
  prediction_sample.png
```

推理脚本同样完全由 `conf/config.yaml` 驱动，不接收命令行配置参数。权重位置由 `training.weight_dir` 和 `training.checkpoint_name` 共同确定，输出目录和保存样本数分别由 `inference.result_dir`、`inference.num_samples` 控制。

# 配置说明

`conf/config.yaml` 分为五部分：

- `common`：设备和随机种子
- `datapipe`：数据文件、样本划分、时间窗口、降采样和 DataLoader
- `model`：隐空间宽度、Fourier 模态、推进次数及归一化配置
- `training`：优化器、学习率调度、教师强制、早停和权重目录
- `inference`：推理结果目录和保存样本数

`model.input_channels` 会根据配置中的 `t_in * out_dim` 自动更新，因此修改历史窗口时无需手工同步模型输入维度。

关键模型参数：

- `op_size`：Koopman 隐空间通道数
- `modes_x`、`modes_y`：两个空间方向保留的 Fourier 模态数
- `decompose`：单次预测中重复应用 Koopman 算子的次数
- `linear_type`：是否使用线性残差推进
- `normalization`：是否在隐空间 shortcut 上使用 BatchNorm

# 数据格式

`.mat` 文件需包含变量 `u`，标准数据形状为：

```text
[1200, 64, 64, 20]
```

| 维度 | 含义 |
| --- | --- |
| `1200` | 独立流场样本数 |
| `64, 64` | 二维规则网格高和宽 |
| `20` | 连续时间步数 |

数据管道输出：

- `pos`：`[H*W, 2]`，归一化二维坐标
- `x`：`[H*W, t_in*out_dim]`，历史状态
- `y`：`[H*W, t_out*out_dim]`，未来状态

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Xiong, W. et al. Koopman Neural Operator as a Mesh-free Solver of Non-linear Partial Differential Equations. Journal of Computational Physics, 2024.
- Xiong, W. et al. KoopmanLab: Machine Learning for Solving Complex Physics Equations. APL Machine Learning, 2023.
- 模型实现源自 GPL-3.0 许可的 KoopmanLab，本模型包沿用 GPL-3.0 许可证并保留来源说明。
