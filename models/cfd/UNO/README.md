
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">UNO</span>
  </strong>
</p>

# 模型介绍

UNO（U-shaped Neural Operator）将 U-Net 的多尺度编码器-解码器结构与 Fourier Neural Operator 的谱卷积结合，在不同空间尺度上学习偏微分方程的解算子，并通过跳跃连接保留局部流场细节。

本模型包面向二维规则网格 Navier-Stokes 时序预测。默认使用前 10 个时间步作为输入，通过自回归方式预测后续 10 个时间步。

论文：U-NO: U-shaped Neural Operators  
https://arxiv.org/abs/2204.11127

# 仓库说明

本仓库是 OneScience 整理的 UNO 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用 YAML 配置训练二维 UNO
- 对 Navier-Stokes 流场执行多步自回归预测
- 计算相对 L2 误差并保存预测张量和可视化结果
- 通过命令行覆盖样本数、时间窗口、空间降采样和模型规模
- 在 CPU、GPU 或 DCU 上运行

当前不支持能力：

- 不内置预训练权重
- 不内置约 394 MiB 的 Navier-Stokes 原始数据文件
- 独立模型仅保留当前案例需要的二维 UNO，不包含原 OneScience 通用 1D/3D 模块组件

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 流场时序预测 | 根据历史涡量场自回归预测未来 Navier-Stokes 状态 |
| 神经算子训练 | 验证 Fourier 谱卷积与 U-shaped 多尺度结构的组合 |
| CFD 快速代理 | 学习规则网格上的历史场到未来场映射 |
| 模型流程验证 | 通过小样本、单 epoch 配置检查训练和推理流程 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 数据、模型、训练及输出配置 | 路径相对于模型包根目录解析 |
| `model/uno.py` | 独立二维 UNO 模型 | 不依赖 `onescience.modules` |
| `scripts/common.py` | 配置、设备、指标和自回归公共函数 | 训练和推理共用 |
| `scripts/train.py` | 训练与验证脚本 | 保存最佳相对 L2 检查点 |
| `scripts/inference.py` | 推理、评估与可视化脚本 | 读取 `weight/*.pt` |
| `data/` | Navier-Stokes 数据目录 | 放置 benchmark `.mat` 文件 |
| `weight/` | 模型权重目录 | 训练时自动写入检查点 |
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
modelscope download --model OneScience/UNO --local_dir ./UNO
cd UNO
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

默认检查点保存至 `weight/uno_navier_stokes.pt`。训练脚本完全由 `conf/config.yaml` 驱动，不接收命令行配置参数。

### 推理、评估和可视化

```bash
python scripts/inference.py
```

脚本默认读取 `weight/uno_navier_stokes.pt`，并在 `result/` 下生成：

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
- `model`：隐藏通道、Fourier 模态、归一化和空间 padding
- `training`：优化器、学习率调度、早停和权重目录
- `inference`：推理结果目录和保存样本数

模型的 `in_dim` 会根据配置中的 `t_in * out_dim` 自动更新，因此修改历史窗口时无需手工同步模型输入维度。

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

- Rahman, M. A., Ross, Z. E., and Azizzadenesheli, K. U-NO: U-shaped Neural Operators. arXiv:2204.11127, 2022.
- Li, Z. et al. Fourier Neural Operator for Parametric Partial Differential Equations. arXiv:2010.08895, 2020.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
