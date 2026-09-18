<p align="center"><strong><span style="font-size: 30px;">WeatherBench2</span></strong></p>

# 模型介绍

WeatherBench2 是面向下一代数据驱动全球天气模型的评估基准。基准同时覆盖确定性、集合概率、偏差和频谱诊断。

论文：WeatherBench 2: A Benchmark for the Next Generation of Data-Driven Global Weather Models  
https://arxiv.org/abs/2308.15560

# 模型描述

该基准由 Google Research、Google DeepMind 和 ECMWF 的研究团队提出。基准使用 ERA5、IFS 以及多种数据驱动模型的 2020 年全球预报数据。基准适用于 1–14 天全球天气预报系统的确定性、概率、偏差和空间尺度评估。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 确定性评估 | 计算 RMSE、ACC、Bias 和 SEEPS。 |
| 概率评估 | 计算 CRPS 和 spread-skill ratio。 |
| 集合诊断 | 比较集合均值、离散度和技巧。 |
| ModelScope/OneCode 运行 | 验证结构化数据、训练、推理、评估和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证紧凑基线训练入口。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/WeatherBench2 --local_dir ./WeatherBench2
cd WeatherBench2
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

WeatherBench2 使用 2020 年全球预报评估 ERA5、IFS 和多种数据驱动系统，并统一重网格至 1.5°。虚拟数据保留 8 个 headline 变量和集合维度，仅减少时间样本和执行网格。该数据只验证基准计算流程，不代表官方排行榜结果。

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

训练紧凑基线并生成单一可恢复 checkpoint，单卡和双进程 DDP 均已验证通过。训练结果保存到：

```text
result/checkpoints/weatherbench2.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。WeatherBench2 是评估基准而非单一预训练模型，因此不存在统一的官方模型权重链接。

### 推理

```bash
python scripts/inference.py
```

推理恢复 checkpoint，并生成 8 成员、8 个样本和 8 个变量的集合预测。集合输出 shape 为 `[8,8,8,24,48]`。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算 RMSE、CRPS 和 spread-skill ratio，并生成空间误差图。指标和 PNG 均通过有效性检查。评估结果保存到：

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

本仓库为 WeatherBench2 公开规格的独立工程复现版本。

WeatherBench 2 评估代码、ERA5、IFS 及各参评模型预报数据分别遵循其来源项目的许可证与数据使用条款。
