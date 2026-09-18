<p align="center">
  <strong><span style="font-size: 30px;">NNCAM</span></strong>
</p>

# 模型介绍

NNCAM 根据大气柱状态预测云、对流和辐射等次网格过程产生的物理倾向与通量，主要用于数据驱动气候模式参数化研究。

论文：Deep learning to represent subgrid processes in climate models  
https://gmd.copernicus.org/articles/11/3999/2018/

# 模型描述

该方法由慕尼黑大学、加州大学欧文分校和哥伦比亚大学的研究团队提出。论文使用 SPCAM 水行星模拟一年的约 1.4 亿个大气柱样本训练网络。模型适用于从 94 维大气柱状态预测 65 维加热、湿化、辐射通量和降水输出。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 次网格过程参数化 | 根据温度、湿度、风和地表强迫预测物理倾向与通量。 |
| 大气柱诊断 | 验证 30 层大气柱的加热、湿化、辐射和降水关系。 |
| 本地工程验证 | 使用结构化虚拟样本验证训练、推理、守恒诊断和可视化流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、参数化指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/NNCAM --local_dir ./NNCAM
cd NNCAM
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活 DTK 及 Conda
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活 Conda
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

论文训练数据来自 SPCAM 水行星模拟，时间步为 30 分钟并包含 30 个垂直层。输入为 `[B,94]` 的温度、湿度、风和地表强迫，目标为 `[B,65]` 的加热、湿化、四个辐射通量及降水。本仓库使用少量具有垂直和物理关联的虚拟样本验证工程流程，不代表 SPCAM 的真实数据分布、训练规模或论文性能。

```bash
python scripts/fake_data.py
```

### 训练

单卡训练可使用：

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

默认配置将网络由论文的 9 层、每层 256 个节点缩小为 4 层、每层 32 个节点，并减少训练轮数，但不缩小 94 维输入、65 维输出和 30 层垂直协议。正式实验需要真实 SPCAM 数据和论文规模模型，训练产物保存到：

```text
result/checkpoints/nncam.pt
result/training/metrics.json
```

### 训练权重

论文未提供可直接加载的官方模型权重，本仓库不在 `weight/` 中内置权重。本地训练生成的 checkpoint 保存到 `result/checkpoints/nncam.pt`，不得描述为官方预训练权重。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，根据完整大气柱状态生成次网格倾向、辐射通量和降水预测。完整数值结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估按输出组计算 RMSE 和 R²，并生成分组误差及降水预测对比图。虚拟数据结果仅用于验证工程流程，不代表论文正式性能；结果保存到：

```text
result/evaluation/metrics.json
result/evaluation/comparison.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 NNCAM 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
