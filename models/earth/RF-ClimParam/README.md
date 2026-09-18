<p align="center">
  <strong><span style="font-size: 30px;">RF-ClimParam</span></strong>
</p>

# 模型介绍

RF-ClimParam 用于从高分辨率大气模拟中学习对流、云微物理、辐射、湍流扩散和地表通量等未解析过程，并在不同水平分辨率的粗网格气候模型中提供稳定的亚网格参数化，主要用于多分辨率气候模拟、参数化尺度依赖研究和降水气候统计重建。

论文：Stable machine-learning parameterization of subgrid processes for climate modeling at a range of resolutions  
https://arxiv.org/abs/2001.03151

# 模型描述

RF-ClimParam 对应的方法由 Massachusetts Institute of Technology 的 Janni Yuval 和 Paul A. O'Gorman 提出。论文使用三维高分辨率 System for Atmospheric Modeling 理想水球模拟的粗粒化状态、瞬时物理倾向、湍流扩散率和地表通量数据训练随机森林。模型适用于多分辨率大气亚网格参数化、粗分辨率气候模拟和平均及极端降水统计评估。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 多分辨率亚网格参数化 | 在 x4/x8/x16/x32 分辨率分别预测柱物理 tendency 与扩散量。 |
| 联合多输出回归 | 在同一树叶中保留跨变量、跨高度输出的联合均值，不拆分输出。 |
| 离线工程评估 | 在四个完整粗网格场上逐尺度、逐输出计算 R2 和 RMSE。 |
| 在线耦合代理 | 在附加的 x32 native `18×48` 粗网格评估 3 h 降水的纬向均值和极端分位数。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证数据生成、训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/RF-ClimParam --local_dir ./RF-ClimParam
cd RF-ClimParam
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行；随机森林本身使用 CPU/NumPy。
- CPU 可用于默认小样本配置的完整连通性验证。
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

```bash
python scripts/fake_data.py
```

本仓库使用少量结构化虚拟大气柱样本验证工程流程，并为 x4、x8、x16 和 x32 分别保留 `144×360`、`72×180`、`36×90` 和 `18×45` 的完整粗网格快照。数据保持真实 48 层及 `145→144`、`62→17` 随机森林接口，只缩小快照数量、训练抽样柱数和树数。虚拟数据包含具有空间和垂直关联的温度、水汽、凝结物、风及通量，仅用于验证多分辨率参数化、训练、推理和评估流程，不代表官方 SAM 数据分布与论文性能。

### 训练

单卡训练可使用：

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练从各分辨率的完整空间场中抽取少量大气柱，分别拟合两类联合多输出随机森林。训练生成四种分辨率的模型参数和训练样本统计，结果保存到：

```text
result/checkpoints/rf_climparam.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。原始论文提供了不同分辨率的随机森林 estimators，权重与模型产物请参考作者给出的 OSF 归档：https://doi.org/10.17605/OSF.IO/36YPT 。

### 推理

```bash
python scripts/inference.py
```

推理结果包含 x4、x8、x16 和 x32 四种分辨率下的真实目标与亚网格过程预测。结果同时包含各尺度的完整空间场、诊断降水和网格位置信息，并附加 x32 原生网格结果。所有数值结果保存到 `result/output/predictions.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估结果包含四种分辨率下亚网格过程的预测表现，以及粗网格降水的在线代理表现。可视化展示不同分辨率的结果对比和目标与预测降水的纬向分布。结构化结果和辅助图片分别保存到 `result/evaluation/metrics.json` 与 `result/evaluation/comparison.png`。虚拟数据结果仅用于验证工程流程，不代表论文的正式 SAM 在线模拟性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 RF-ClimParam 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
