<p align="center"><strong><span style="font-size: 30px;">RainRunoff-LSTM</span></strong></p>

# 模型介绍

RainRunoff-LSTM 使用长期气象历史预测日尺度流量。模型支持单流域、区域共享和区域预训练后微调实验。

论文：Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks  
https://doi.org/10.5194/hess-22-6005-2018

# 模型描述

该模型由 University of Natural Resources and Life Sciences Vienna 的研究团队提出。模型使用 CAMELS Daymet 的降水、最低与最高温度、短波辐射、蒸汽压以及流量观测训练。模型适用于日尺度降雨径流模拟、跨流域区域模型和预训练微调实验。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 日流量模拟 | 从 365 日气象强迫预测下一日流量。 |
| 区域模型 | 在同一水文区内共享模型参数。 |
| 预训练微调 | 使用区域模型初始化单流域模型。 |
| ModelScope/OneCode 运行 | 验证数据、训练、推理、水文指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

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

论文使用 241 个 CAMELS 流域、五个 Daymet 日气象变量和 365 日历史窗口。虚拟数据保持 `[B,365,5]` 输入和下一日流量目标，仅减少流域与样本数量。结果仅用于工程验证。

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
torchrun --standalone --nproc_per_node=2 scripts/train.py
```

训练使用 MSE 优化两层 LSTM，单卡与双进程 DDP 均已通过。训练结果保存到：

```text
result/checkpoints/rainrunoff_lstm.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。论文未发布可直接加载的官方预训练权重，因此不提供权重链接。

### 推理

```bash
python scripts/inference.py
```

推理恢复 checkpoint 并输出 12 个虚拟流域的日流量标量，shape 与有限数值检查均已通过。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算 NSE 并生成流量预测与真值对比图。指标与 PNG 均通过有效性检查，评估结果保存到：

```text
result/evaluation/metrics.json
result/evaluation/comparison.png
```

# 引用与许可证

本仓库为 RainRunoff-LSTM 公开规格的独立工程复现版本。

原始论文采用 CC BY 4.0 许可证；原始论文、相关代码和 CAMELS 数据仍应按照各自项目的许可证及使用条款使用。
