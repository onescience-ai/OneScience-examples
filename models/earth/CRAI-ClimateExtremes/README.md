<p align="center">
  <strong><span style="font-size: 30px;">CRAI-ClimateExtremes</span></strong>
</p>

# 模型介绍

CRAI-ClimateExtremes 根据不完整的气候极端指数场及其有效掩码重建缺测区域，主要用于历史气候极端事件分析和稀疏观测资料重建。

论文：Artificial intelligence reveals past climate extremes by reconstructing historical records  
https://doi.org/10.1038/s41467-024-53464-2

# 模型描述

该方法由德国气候计算中心、英国气象局和汉堡大学等机构的研究团队提出。论文使用 8 个 CMIP6 模式的 45 个历史模拟、HadEX-CAM 观测资料以及 ERA5 等再分析数据。模型适用于重建 TX90p、TN90p、TX10p 和 TN10p 月尺度温度极端指数。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 气候极端指数重建 | 从不完整指数场和有效掩码重建四类月尺度温度极端指数。 |
| 不规则缺测处理 | 验证 partial convolution、掩码传播和缺失区域损失。 |
| 本地工程验证 | 使用完整 `144×192` 网格的结构化虚拟数据验证端到端流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、气候重建指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/CRAI-ClimateExtremes --local_dir ./CRAI-ClimateExtremes
cd CRAI-ClimateExtremes
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

论文使用 1901-2014 年 CMIP6 月尺度极端指数，并应用 HadEX-CAM 缺测掩码构造训练样本。输入为指数场与有效掩码 `[B,2,144,192]`，目标为 `[B,1,144,192]` 的完整指数场。本仓库使用少量结构化虚拟样本验证工程流程，不代表 CMIP6 或 HadEX-CAM 的真实数据分布、训练规模或论文性能。

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

默认配置仅缩小样本数、网络宽度、训练轮数和 ensemble 成员数，不缩小 `144×192` 空间网格和四类指数协议。正式实验需要真实 CMIP6 与 HadEX-CAM 数据和论文规模训练配置，训练产物保存到：

```text
result/checkpoints/crai_climateextremes.pt
result/training/metrics.json
```

### 训练权重

论文未提供可直接加载的官方模型权重，本仓库不在 `weight/` 中内置权重。本地训练生成的 checkpoint 保存到 `result/checkpoints/crai_climateextremes.pt`，不得描述为官方预训练权重。

### 推理

```bash
python scripts/inference.py
```

推理加载统一 ensemble checkpoint，重建缺测极端指数并保存成员均值和离散度。完整数值结果保存到：

```text
result/output/predictions.npz
result/output/metadata.json
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算缺失区域 RMSE、Spearman 相关系数、bias 和邻域空间相关，并生成目标、输入和重建结果对比图。虚拟数据结果仅用于工程验证，不代表论文正式性能；结果保存到：

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

本仓库为 CRAI-ClimateExtremes 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
