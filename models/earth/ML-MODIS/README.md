<p align="center">
  <strong><span style="font-size: 30px;">ML-MODIS</span></strong>
</p>

# 模型介绍

ML-MODIS 复现 Chen 等（2022）提出的卫星机器学习流程，依据气象状态估计未受 2014 年火山气溶胶扰动的云属性反事实，并据此诊断气溶胶云效应及其短波辐射贡献。

论文：Machine learning reveals climate forcing from aerosols is dominated by increased cloud cover  
https://doi.org/10.1038/s41561-022-00991-6

# 模型描述

该方法由 University of Exeter、Met Office、ETH Zurich、University of Cambridge、NASA Goddard Space Flight Center、University of Leeds 和 Ludwig Maximilian University of Munich 等机构组成的研究团队提出。论文将 MODIS Collection 6.1 云产品与 ERA5 气象数据配对，使用非 2014 年样本训练月份与云属性目标相互独立的随机森林。模型根据气象条件生成反事实云属性，再将 2014 年 MODIS 观测与反事实比较，以诊断火山气溶胶引起的云响应和短波辐射强迫贡献。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 云属性反事实预测 | 从 ERA5 气象状态估计无火山扰动条件下的 `Nd`、`reff`、`LWP` 和 `CF`，并与 2014 年观测比较。 |
| OOB 解释与辐射诊断 | 计算 OOB 技能和置换重要性，并诊断 Twomey、LWP 与 CF 的相对短波辐射贡献。 |
| 本地流程验证 | 使用结构化虚拟数据验证数据生成、训练、checkpoint 恢复、推理、评估和可视化。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、反事实与辐射诊断指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/ML-MODIS --local_dir ./ML-MODIS
cd ML-MODIS
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

论文使用 MODIS Collection 6.1 云产品和 ERA5 气象数据，按年月、Terra/Aqua 平台及经纬度对齐，并排除 2014 年样本进行训练。每个样本包含 114 个输入特征，即 9 个廓线变量在 10 个压力层上的 90 个字段和 24 个单层字段，目标为 `Nd`、`reff`、`LWP` 和 `CF`。虚拟数据包含合理的气象、时空和云属性关系，仅用于验证工程流程，不代表 MODIS 或 ERA5 的真实数据分布、训练规模或论文性能。

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

训练为 September/October 的四个云属性分别构建随机森林；默认配置将每个森林由论文的 100 棵树缩为 12 棵，但不缩小 114 维输入、月份、目标和多模态对齐协议。正式实验需要真实 MODIS、ERA5 数据并核验补充材料中的字段定义，训练产物保存到：

```text
result/checkpoints/ml_modis.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重，也未发现论文作者公开的模型权重或可直接下载的 checkpoint。论文仅说明代码可向通讯作者合理申请，当前本地 checkpoint 是工程训练产物，不声明与作者未公开的权重兼容。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，根据 ERA5 气象条件生成四种云属性的反事实预测，并保存逐树预测、森林均值、观测和对齐信息。完整数值结果保存到 `result/output/predictions.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估计算 OOB MSE、R²、Pearson 相关系数、置换重要性、2014 年面积加权响应、susceptibility 和相对短波辐射贡献。结果保存到 `result/evaluation/metrics.json` 和 `result/evaluation/comparison.png`；虚拟数据结果仅用于工程验证，不代表论文正式性能或气候归因结论。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 ML-MODIS 公开规格的独立工程复现版本。

本仓库代码采用 Apache-2.0 许可证；原论文和作者代码仍受其各自版权及使用条款约束。

MODIS Collection 6.1 和 ERA5 数据的使用应分别遵循 NASA Earthdata/LAADS DAAC 和 Copernicus Climate Data Store/ECMWF 的现行条款。
