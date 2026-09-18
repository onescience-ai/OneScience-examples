<p align="center"><strong><span style="font-size: 30px;">WeatherBench</span></strong></p>

# 模型介绍

WeatherBench 是面向数据驱动全球天气预报的数据集与评估基准。本复现实现论文实际使用的五层全卷积 CNN，并完成 3 天和 5 天 Z500/T850 预报评估。

论文：WeatherBench: A Benchmark Data Set for Data-Driven Weather Forecasting  
https://arxiv.org/abs/2002.00469

# 模型描述

该基准由慕尼黑工业大学、ECMWF、Stockholm University、University of Washington 和 University of Toronto 团队提出。基准使用 1979–2018 年 ERA5 再分析数据，提供多分辨率全球场和 13 个压力层。论文示例 CNN 适用于直接或迭代预测 Z500 和 T850，并与 persistence、climatology、线性回归和 IFS 比较。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 中期天气预报 | 评估 3 天和 5 天 Z500/T850。 |
| 基线比较 | 对比 persistence 和 climatology。 |
| 纬度加权评估 | 计算 WeatherBench RMSE 和 ACC。 |
| ModelScope/OneCode 运行 | 验证结构化数据、训练、推理、天气指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/WeatherBench --local_dir ./WeatherBench
cd WeatherBench
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

WeatherBench 原始 ERA5 数据包含 8 个三维变量、13 个压力层、6 个单层变量和常量场，并提供最高 `128×256` 的处理网格。论文 CNN 实验在 `32×64` 网格使用 Z500 和 T850 两通道，虚拟数据完整保留该模型输入输出与 6 小时迭代协议，仅减少样本和隐藏通道。结果仅用于工程验证。

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

训练已完成单卡和双进程 DDP 验证，并生成单一可恢复 checkpoint。五层全卷积 CNN 已使用六小时 Z500/T850 目标完成 MSE 优化和参数更新。训练结果保存到：
```text
result/checkpoints/weatherbench_cnn.pt
result/training/metrics.json
```

### 训练权重

论文未提供可供本仓库直接加载的官方 CNN 权重，本地 checkpoint 仅用于工程验证。

### 推理

```bash
python scripts/inference.py
```

推理将六小时模型迭代至 3 天和 5 天，并保留真值、初值和坐标。输出 shape 为 `[4,2,2,32,64]` 且数值有限。恢复后的 checkpoint 能够连续完成 12 步和 20 步迭代，并保持经度周期边界协议。推理结果保存到：
```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算纬度加权 RMSE、ACC 和 persistence RMSE，并生成 5 天 Z500 空间误差图。指标和 PNG 均通过有效性检查。评估结果保存到：
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

本仓库为 WeatherBench 公开规格的独立工程复现版本。

原始论文、WeatherBench 代码和 ERA5 数据仍应按照各自项目的许可证及使用条款使用。
