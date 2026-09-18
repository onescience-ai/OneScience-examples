<p align="center">
  <strong><span style="font-size: 30px;">GlobalSurgeML</span></strong>
</p>

# 模型介绍

GlobalSurgeML 使用验潮站周围的风场、平均海平面气压、海表温度和降水信息模拟每日最大风暴增水。模型先以 PCA 消除网格预测变量的多重共线性，再使用逐步多元线性回归或随机森林建立站点级映射，输出单位为米的每日最大非潮汐残差，适合低成本的长时间尺度和大空间尺度风暴潮模拟。

论文：Data-Driven Modeling of Global Storm Surges  
https://doi.org/10.3389/fmars.2020.00260

# 模型描述

该方法由 University of Central Florida 与 Universidad de Cantabria 的研究人员提出。论文使用 GESLA-2 验潮站、CCMP、20CRV2c、Microwave OI SST、GPCP、ERA-Interim 和 GTSR 数据训练与验证。模型适用于准全球验潮站的每日最大风暴增水回归、极端事件评估以及与 GTSR 水动力再分析比较。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 日最大增水模拟 | 从站点周边气象与海洋预测变量的 PCA 特征估计每日最大非潮汐残差。 |
| 滞后效应建模 | 使用 6 小时输入并考虑风暴潮发生前最多 30 小时的风速与气压信息。 |
| 方法对比 | 比较逐步线性回归、随机森林、遥感输入与 ERA-Interim 输入的六种配置。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动线性模型的分布式数据并行优化。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/GlobalSurgeML --local_dir ./GlobalSurgeML
cd GlobalSurgeML
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

本仓库使用少量虚拟站点日样本验证工程流程。

虚拟数据用具有持续性的天气状态、季节周期、纬度效应和非线性气旋强迫共同产生相关特征与增水目标，不是无关联随机噪声。它保持论文表 1 给出的约 50/300 维配置；论文同时指出原始局地网格最多约 5000 个变量，保留 90% 方差后通常为 300 至 500 个 PC，因此并不存在适用于所有站点的唯一固定原始空间张量尺寸。虚拟数据仅用于验证核心方法、训练、推理和评估流程，不代表官方数据分布与训练规模。

```bash
python scripts/fake_data.py
```

### 训练

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

默认工程配置保持四类输入的 50/300 维和单值输出不变，仅缩小样本数、线性回归最多保留的特征数和随机森林深度。正式实验应重新执行逐站点 PCA、10 折交叉验证和六配置多数指标选择，并使用完整真实记录。

```text
result/checkpoints/globalsurgeml.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。论文及其补充材料未提供可确认的官方模型 checkpoint；当前 checkpoint 是本独立 PyTorch/scikit-learn 工程格式，不声明兼容任何外部权重。

### 推理

```bash
python scripts/inference.py
```

推理加载 `globalsurgeml.pt`，用四组站点日 PCA 输入恢复并运行六种模型配置。完整数值结果包含六组日最大增水预测、观测目标、GTSR 基线、时间和站点坐标，保存到 `result/output/predictions.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估按论文协议计算 Pearson 相关系数、RMSE、NSE 和相对 RMSE，并对观测值第 95 百分位以上的极端增水单独评估，同时按纬度区分热带与副热带/温带样本。脚本保存 `result/evaluation/metrics.json`，并生成增水序列及观测-模拟散点图 `result/evaluation/comparison.png`。虚拟数据结果仅用于验证工程流程，不代表论文真实测试集性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 GlobalSurgeML 论文公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
