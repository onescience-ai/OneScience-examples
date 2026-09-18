<p align="center">
  <strong><span style="font-size: 30px;">IceNet</span></strong>
</p>

# 模型介绍

IceNet 面向季节尺度北极海冰预测，根据历史海冰和气候变量生成未来 6 个月的逐月概率预报，为海冰预测方法研究提供可运行的工程验证流程。

论文：Seasonal Arctic sea ice forecasting with probabilistic deep learning  
https://doi.org/10.1038/s41467-021-25257-4

# 模型描述

该方法由英国南极调查局（British Antarctic Survey）、艾伦·图灵研究所（Alan Turing Institute）、伦敦大学学院（UCL）等机构的研究团队提出。论文使用 OSI-SAF 海冰浓度观测、ERA5 再分析和 CMIP6 气候模式模拟数据开展训练。模型执行未来 6 个月逐月北极海冰浓度类别概率预测任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 季节海冰概率预报 | 预测未来 6 个月无冰、边缘冰和密集冰三类概率。 |
| 核心方法验证 | 验证单次多时效预测、掩膜加权 categorical focal loss、ensemble 和温度缩放。 |
| 本地工程验证 | 使用结构化虚拟数据验证固定网格的数据契约、推理和逐月海冰指标。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理和结构化评估流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/IceNet --local_dir ./IceNet
cd IceNet
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

论文使用 OSI-SAF 海冰浓度观测、ERA5 再分析和 CMIP6 气候模式模拟数据。
本仓库的数据契约采用 `50×432×432` 输入，并生成未来 6 个月、每月 3 类的 `6×3×432×432` 输出。
`scripts/fake_data.py` 生成的结构化虚拟数据仅用于验证工程流程，不代表真实数据分布、训练规模或论文性能。

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

默认工程配置使用逐月掩膜加权 categorical focal loss，并在验证集上拟合温度缩放参数，同时缩小 ensemble 成员数和模型基础通道宽度，但不缩小固定的 `50×432×432` 输入与 `6×3×432×432` 输出维度。
训练生成本地 checkpoint 和结构化指标，产物保存到：

```text
result/checkpoints/icenet.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置官方权重。论文的 25 个网络权重、2012 至 2020 年预报及结果数据可从 NERC EDS UK Polar Data Centre 获取：https://doi.org/10.5285/71820E7D-C628-4E32-969F-464B7EFB187C 。

### 推理

```bash
python scripts/inference.py
```

推理加载本地训练 checkpoint，输出逐月三分类概率、海冰存在概率和预测类别：

```text
result/output/predictions.npz
result/output/metadata.json
```

### 评估和可视化

```bash
python scripts/result.py
```

评估逐月计算三分类准确率、二分类准确率、海冰范围误差、综合冰缘误差、冰缘覆盖率和概率校准分箱。脚本仅生成结构化评估结果 `result/evaluation/metrics.json`，不生成可视化图像文件。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 IceNet 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
