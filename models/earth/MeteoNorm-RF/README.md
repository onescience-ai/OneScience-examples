<p align="center"><strong><span style="font-size: 30px;">MeteoNorm-RF</span></strong></p>

# 模型介绍

MeteoNorm-RF 使用随机森林和气象重采样分离北京空气污染变化中的天气影响与长期排放趋势，用于清洁空气行动效果评估。模型保留季节、日变化和站点差异，可同时分析六种主要污染物的气象归一化变化。

论文：*Assessing the impact of clean air action on air quality trends in Beijing using a machine learning technique*  
DOI：https://doi.org/10.5194/acp-19-11303-2019

# 模型描述

该方法由伯明翰大学、中国科学院等机构的研究团队提出。论文使用北京 12 个国家监测站的 6 种污染物小时数据和机场气象观测。随机森林通过重采样相同小时和相邻季节窗口内的天气条件，估计气象归一化污染浓度。模型适用于气象归一化污染浓度和 Theil-Sen 长期趋势分析。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 气象归一化 | 通过天气重采样估计固定时间下的归一化污染浓度。 |
| 空气质量趋势 | 计算 6 种污染物的稳健长期趋势。 |
| 本地工程验证 | 保持 12 站、6 污染物、小时频率和完整特征。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证数据、训练、推理、空气质量指标和可视化。 |
| 多卡训练 | 通过 `torchrun` 验证分布式任务并行和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/MeteoNorm-RF --local_dir ./MeteoNorm-RF
cd MeteoNorm-RF
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

本仓库使用少量虚拟样本验证工程流程，虚拟数据包含 12 个监测站、6 种污染物、21 个时间/气象/站点特征、小时关系和完整表格维度。虚拟数据保持真实站点数、污染物数、输入特征和小时频率，仅将时间范围缩短为 30 天，并保留排放、交通、天气和站点相关结构。该数据仅用于验证随机森林、气象归一化、训练、推理和评估流程，不代表官方观测分布与训练规模。

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

训练使用固定 70/30 数据划分拟合六输出随机森林，并支持按树编号进行多进程任务并行。默认配置将论文的数百棵树缩小为 12 棵，但不缩小站点、污染物和输入维度。训练产物保存到：

```text
result/checkpoints/meteonorm_rf.pkl
result/training/metrics.json
```

### 训练权重

论文未提供预训练模型权重，本仓库不在 `weight/` 中内置权重。

### 推理

```bash
python scripts/inference.py
```

推理加载本地随机森林 checkpoint，以时间、气象和站点特征生成六种污染物浓度预测。随后在相同小时和相邻季节窗口内重采样天气条件，形成气象归一化预测集合。结果保留普通预测、归一化均值和时空索引。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估按污染物计算 RMSE、R²、FAC2、MB、MGE、NMB、NMGE、COE 和 IOA，并比较观测与气象归一化浓度。结果还包含六种污染物的 Theil-Sen 稳健趋势，并生成污染物预测表现和归一化趋势对比图。虚拟数据结果仅用于工程验证，不代表论文正式性能。评估结果保存到：

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

本仓库为 MeteoNorm-RF 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY 4.0 许可证；论文、官方观测数据和第三方资源仍应按照各自项目的许可证及使用条款使用。
