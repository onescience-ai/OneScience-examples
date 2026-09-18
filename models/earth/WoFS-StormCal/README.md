<p align="center"><strong><span style="font-size: 30px;">WoFS-StormCal</span></strong></p>

# 模型介绍

WoFS-StormCal 用于解决短时风暴尺度强天气概率预报的校准问题，综合集合风暴轨迹中的风暴状态、周边环境和对象形态信息，判断风暴产生龙卷风、严重冰雹或严重大风的可能性。模型主要用于提高集合预报概率的可靠性，为临近预报、强天气风险研判和预报员决策提供概率指导。

论文：Using Machine Learning to Calibrate Storm-Scale Probabilistic Guidance of Severe Weather Hazards in the Warn-on-Forecast System  
https://arxiv.org/abs/2012.00679

# 模型描述

WoFS-StormCal 由 University of Oklahoma、Cooperative Institute for Mesoscale Meteorological Studies 和 NOAA National Severe Storms Laboratory 的研究团队提出。论文使用 2017 至 2019 年 NOAA Hazardous Weather Testbed Spring Forecasting Experiments 的 WoFS 集合预报和本地风暴报告训练与验证。模型适用于龙卷风、严重冰雹和严重大风的短时风暴尺度概率预报与校准。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 风暴尺度概率预报 | 根据集合风暴轨迹对象预测龙卷风、严重冰雹和严重大风概率。 |
| 概率校准 | 使用交叉验证概率拟合单调 isotonic 映射。 |
| 分时效建模 | 分别处理 first hour 和 second hour 的 30 分钟风暴轨迹窗口。 |
| ModelScope/OneCode 运行 | 验证训练、推理、概率评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/WoFS-StormCal --local_dir ./WoFS-StormCal
cd WoFS-StormCal
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

虚拟数据使用组织度、旋转、热力不稳定度、风切变、冷池和集合离散度等潜变量构造组内及跨组相关结构，并保持真实 113 维不缩减。该数据仅用于验证特征加载、概率训练、校准、推理和评估流程，不代表真实强天气样本分布与论文性能。

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

训练输出保存到：

```text
result/checkpoints/wofsstormcal.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。论文未提供可确认的官方 checkpoint，当前工程 checkpoint 不声明兼容外部权重。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，根据 113 维对象特征输出三个灾种的校准概率，并保存到 `result/output/predictions.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估结果包含各灾种和时效组的概率技巧、分类技巧与可靠性指标，并保存到 `result/evaluation/metrics.json`。脚本同时生成性能图和可靠性图。虚拟数据结果仅用于验证工程流程，不代表论文真实测试集性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 WoFS-StormCal 论文公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
