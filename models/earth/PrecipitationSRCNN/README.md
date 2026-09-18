<p align="center">
  <strong><span style="font-size: 30px;">PrecipitationSRCNN</span></strong>
</p>

# 模型介绍

PrecipitationSRCNN 用于将低分辨率逐日降水重建为具有更精细空间结构的高分辨率降水场，可作为动力降尺度的低成本补充，主要服务于区域降水降尺度和极端降水空间结构分析。

论文：Complementing Dynamical Downscaling With Super-Resolution Convolutional Neural Networks  
https://doi.org/10.1029/2024GL111828

# 模型描述

该方法由 Oak Ridge National Laboratory、Lawrence Berkeley National Laboratory 和 University of California, Berkeley 的研究团队提出。论文使用 ERA5 逐日降水、ERA5 动力降尺度数据 ERA5DD 和高程数据训练与评估。模型适用于逐日降水空间超分辨率、动力降尺度代理和降水气候统计重建。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 逐日降水降尺度 | 将低分辨率降水和高程条件转换为目标网格降水场。 |
| SRCNN 超分辨率 | 验证模型外上采样和三层 SRCNN 空间重建方法。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证虚拟数据生成、训练、推理、论文指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/PrecipitationSRCNN --local_dir ./PrecipitationSRCNN
cd PrecipitationSRCNN
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

本仓库使用少量逐日虚拟样本验证工程流程，输入为低分辨率降水和高程，目标为同一日期、`216×488` 网格上的高分辨率降水，单位为 `mm/day`。虚拟数据保持真实目标网格和逐日对应关系，只减少样本数量、模型宽度和训练轮次。该数据仅用于验证 SRCNN 的训练、推理和评估流程，不代表 ERA5 或 ERA5DD 的官方数据分布与训练规模。

```bash
python scripts/fake_data.py
```

该命令通过 `scripts/fake_data.py` 生成 `data/daily_precipitation.npz`。

### 训练

单卡训练可使用：

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

默认配置减少了样本数量、模型宽度和训练轮数，但保持 `216×488` 目标网格不变。训练产物保存到：

```text
result/checkpoints/precipitationsrcnn.pt
result/training/metrics.json
```

### 训练权重

当前交付不包含官方 checkpoint。本仓库训练流程会生成工程 checkpoint，但本项目未复制或打包官方权重，也不声明该工程权重属于或等同于官方权重。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，根据低分辨率逐日降水和同网格高程生成 `216×488` 高分辨率降水场。完整数值结果保存到 `result/output/predictions.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估按照论文降水协议计算平均降水、逐年 P95、湿日、极端日、偏差和空间相关，结果保存到 `result/evaluation/metrics.json`；该任务不是多步预报，因此不保存逐时效结果，而是按日期和年份聚合。评估同时生成输入、目标、预测和误差的降水场对比图；虚拟数据结果仅用于工程验证，不代表论文正式性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 PrecipitationSRCNN 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
