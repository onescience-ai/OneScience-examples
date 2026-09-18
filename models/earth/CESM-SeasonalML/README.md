<p align="center">
  <strong><span style="font-size: 30px;">CESM-SeasonalML</span></strong>
</p>

# 模型介绍

CESM-SeasonalML 使用气候模式大集合训练可解释机器学习模型，分别预测 NDJ（11月至次年1月）和 JFM（1月至3月）美国西部标准化季节降水异常的四种大尺度空间型。

论文：Training machine learning models on climate model output yields skillful interpretable seasonal precipitation forecasts  
https://doi.org/10.1038/s43247-021-00225-4

# 模型描述

该方法由加州大学圣地亚哥分校斯克里普斯海洋研究所西部天气与水极端事件中心（CW3E）和加州理工学院 NASA 喷气推进实验室（JPL）的研究团队提出。论文使用 CESM-LENS 进行训练，并使用 ERSSTv5、ERA5 和 CPC Unified Gauge-Based Analysis of Global Daily Precipitation 观测与再分析数据进行测试。模型分别预测 NDJ 和 JFM 美国西部四类降水空间型，并利用随机森林解释关键海洋和大气预测因子的贡献。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 季节降水分类 | 分别预测 NDJ 和 JFM 美国西部四类大尺度降水空间型。 |
| KMeans 目标构建 | 对标准化季节降水场聚类并生成稳定的四类训练目标。 |
| RF 解释分析 | 使用 permutation importance、mean minimum depth 和 root frequency 分析关键预测因子。 |
| 本地流程验证 | 使用结构化虚拟数据验证数据生成、训练、推理、评估和可视化流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、季节分类指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/CESM-SeasonalML --local_dir ./CESM-SeasonalML
cd CESM-SeasonalML
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行神经网络训练。
- CPU 可用于虚拟数据生成以及默认小样本配置的流程验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活 DTK 及 Conda
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
pip install numpy pyyaml matplotlib
```

**GPU环境**

```bash
# 请首先激活 Conda
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
pip install numpy pyyaml matplotlib
```

### 训练数据介绍

论文使用 CESM-LENS 训练，并使用 ERSSTv5、ERA5 和 CPC 数据测试。
RF/XGB、NN 和 LSTM 分别使用 `103`、`416` 和 `28` 维输入，预测四类季节降水空间型目标。
论文未提供完整特征 manifest 和预处理后的最终网格尺寸，因此本仓库的结构化占位特征与 `20×24` 网格均为工程假设。
虚拟数据仅用于验证工程流程，不代表真实数据分布、数据规模或论文性能。

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
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py --models nn lstm
```

默认工程配置训练 RF、XGB 和 LSTM，并缩小树数、训练轮数和 epoch，但不缩小 `103/416/28` 维输入和四个目标类别。
正式数据实验应使用 CESM-LENS 训练数据以及 ERSSTv5、ERA5、CPC 测试数据，并恢复 `conf/config.yaml` 中 `paper_model` 的论文规模配置，训练产物保存到：

```text
result/checkpoints/cesm_seasonal_ml.pt
result/training/metrics.json
```

### 训练权重

本仓库不内置论文权重，公开资源中也没有可确认的官方 checkpoint；论文说明原始代码可向通讯作者索取。运行训练脚本会在 `result/checkpoints/cesm_seasonal_ml.pt` 生成包含模型、KMeans 状态、数据规格和训练记录的本地工程 checkpoint，该文件不代表论文官方预训练权重或论文数值结果。

### 推理

```bash
python scripts/inference.py
```

推理加载本地训练 checkpoint，并校验版本、季节、manifest 和数据形状；所有启用模型的四类概率、预测类别、目标类别、聚类质心、年份、经纬度和目标降水场保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算季节四分类指标和基线，并生成模型、基线及降水空间型对比图。
虚拟数据结果仅用于工程验证，不代表论文性能。

```text
result/evaluation/metrics.json
result/evaluation/comparison.png
result/evaluation/precipitation_clusters.png
result/evaluation/seasonal_predictions.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 CESM-SeasonalML 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
