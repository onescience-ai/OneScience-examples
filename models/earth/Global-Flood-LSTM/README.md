<p align="center"><strong><span style="font-size: 30px;">Global-Flood-LSTM</span></strong></p>

# 模型介绍

Global-Flood-LSTM 是面向全球无资料流域的概率河流流量预报模型。模型根据历史气象、预报强迫和流域属性生成未来 7 天流量分布，并重点评估极端洪水可靠性。

论文：Global prediction of extreme floods in ungauged watersheds  
https://doi.org/10.1038/s41586-024-07145-1

# 模型描述

该模型由 Google Research、欧洲中期天气预报中心、Helmholtz Centre for Environmental Research 和 RAND Corporation 的研究团队提出。模型使用 5,680 个 GRDC 测站的流量观测，以及 HRES、ERA5-Land、CPC、IMERG 和 HydroATLAS 数据训练。模型通过 encoder-decoder LSTM 融合历史与预报强迫并输出概率分布，适用于全球无资料流域的 7 天流量与极端洪水预报。

# 适用场景

| 场景 | 说明 |
|---|---|
| 无资料流域预报 | 使用跨流域共享模型预测未参与训练的流域。 |
| 极端洪水预警 | 评估 precision、recall 和 F1。 |
| 概率流量预报 | 输出随时效变化的条件分布。 |
| 水文基线比较 | 计算 RMSE 和 KGE。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、概率降水指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/Global-Flood-LSTM --local_dir ./Global-Flood-LSTM
cd Global-Flood-LSTM
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

论文使用 365 天历史气象序列、7 天预报强迫、HydroATLAS 静态属性和 GRDC 日流量，覆盖 5,680 个流域。虚拟数据保留 365 天历史、14 个动态源通道、7 日输出和概率分布协议，仅减少流域数、隐藏宽度和训练次数；静态属性数量因论文未列全而明确标为工程账本。结果仅验证工程流程，不代表论文性能。

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

默认虚拟数据训练能够完成非对称 Laplace 负对数似然优化，单卡与双进程 DDP 流程均已验证通过。训练完成后生成可恢复的单一 checkpoint，并记录训练损失。训练结果保存到：

```text
result/checkpoints/global_flood_lstm.pt
result/training/metrics.json
```

### 训练权重

论文说明研究版本在 NeuralHydrology 中实现，但本文未确认具有独立明确许可证的论文原始权重下载地址，因此不提供权重链接。

### 推理

```bash
python scripts/inference.py
```

推理恢复单一 checkpoint，并生成三个模型成员的未来 7 天流量预测。默认输出覆盖 12 个虚拟流域，预测维度和有限数值检查均已通过。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算逐时效 RMSE、极端事件 precision、recall、F1 和总体 KGE，并绘制误差与极端事件技巧。所有指标均为有限数值；虚拟数据使用分位数代理阈值，不等同于论文 Bulletin 17B 重现期阈值。评估结果保存到：

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

本仓库为 Global-Flood-LSTM 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY 4.0 许可证；论文以及 GRDC、HRES、ERA5-Land、CPC、IMERG 和 HydroATLAS 数据仍应按照各自许可证及使用条款使用。
