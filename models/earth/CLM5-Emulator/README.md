<p align="center"><strong><span style="font-size: 30px;">CLM5-Emulator</span></strong></p>

# 模型介绍

CLM5-Emulator 使用机器学习模拟 Community Land Model version 5 的全球生物物理响应。模型根据六个生物物理参数预测 GPP 和 LHF 的 EOF 主成分，并支持有界参数估计。

论文：A machine learning approach to emulation and biophysical parameter estimation with the Community Land Model, version 5  
https://doi.org/10.5194/ascmo-6-223-2020

# 模型描述

该模型由 NCAR 和 CERFACS 的研究团队提出。模型使用 GSWP3 驱动的 CLM5 参数扰动集合以及 FLUXNET-MTE 观测目标训练。模型通过两个独立前馈网络模拟 GPP 和 LHF 的空间主成分，适用于快速代理建模和生物物理参数估计。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| CLM5 代理建模 | 从六个参数预测 GPP 和 LHF 主成分。 |
| 参数估计 | 在归一化参数空间中搜索观测匹配解。 |
| 空间场重建 | 从 EOF 主成分重建全球响应场。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/CLM5-Emulator --local_dir ./CLM5-Emulator
cd CLM5-Emulator
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
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

论文使用 100 个六参数 CLM5 PPE 成员，将五年平均 GPP 和 LHF 全球场压缩为各 3 个 EOF 主成分。虚拟数据保持 `[B,6]` 输入、两个独立目标、3 个模态和 4°×5° 逻辑网格，仅减少成员数、隐藏宽度和训练轮数。结果仅验证工程流程，不代表论文性能。

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
torchrun --nproc_per_node=2 --nnodes=1 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练完成两个目标的联合优化并生成单一可恢复 checkpoint，同时记录 MSE。GPP 与 LHF 两个独立网络均参与反向传播，单卡和双进程 DDP 训练均已通过。训练结果保存到：

```text
result/checkpoints/clm5_emulator.pt
result/training/metrics.json
```

### 训练权重

论文未提供可供本仓库直接加载的官方预训练权重，本地 checkpoint 仅用于工程验证。

### 推理

```bash
python scripts/inference.py
```

推理恢复 checkpoint，预测测试参数的 GPP/LHF 主成分和空间场，并执行有界参数搜索。主成分输出 `[8,2,3]` 和重建场 `[8,2,46,72]` 均通过 shape 与有限数值检查。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算主成分 `R²`、空间 RMSE 和参数估计代价，并生成主成分散点及空间误差图。各项指标均为有限数值，PNG 已通过格式与有效像素检查；虚拟结果不代表论文正式性能。评估结果保存到：

```text
result/evaluation/metrics.json
result/evaluation/comparison.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 CLM5-Emulator 公开规格的独立工程复现版本。

原始论文采用 CC BY 4.0 许可证；论文、CLM5、GSWP3 和 FLUXNET-MTE 数据仍应按照各自项目的许可证及使用条款使用。
