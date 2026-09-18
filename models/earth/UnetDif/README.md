<p align="center">
  <strong><span style="font-size: 30px;">UnetDif</span></strong>
</p>

# 模型介绍

UnetDif 使用论文的偏差目标多任务 U-Net 订正短期强降水预报。

论文：Bias-targeted deep learning enhances short-range heavy rainfall forecasts  
https://doi.org/10.1038/s41612-026-01366-z

# 模型描述

UnetDif 对应的方法由浙江省气象台、国家气象中心和浙江大学的研究团队提出。论文使用欧洲中期天气预报中心（ECMWF）预报场、中国气象局多源降水分析产品（CMPA）和地形数据构建长三角区域样本。模型以 `ECMWF - CMPA` 降水偏差为学习目标，通过共享 U-Net 主干和干区、虚警区、正偏差、负偏差四个输出头完成 3 小时降水订正。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 3 小时降水偏差订正 | 在固定长三角网格上由 39 通道输入生成 ECMWF 订正降水场。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证虚拟数据生成、六项损失训练、推理、评估和可视化流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中运行完整工程流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/UnetDif --local_dir ./UnetDif
cd UnetDif
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

训练数据包含 ECMWF 气象预报、CMPA 降水观测和静态地形。每个样本使用 39 个输入通道，目标为对应时段的 CMPA 降水和 ECMWF 降水偏差。数据采用长三角 `0.125°` 分辨率的 `56×56` 网格。每个雨日包含 8 个连续 3 小时时段，降水单位为 `mm/3h`。虚拟数据仅用于验证训练、推理和评估流程，不代表 ECMWF、CMPA 和真实地形的数据分布或论文正式性能。

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

训练完成干区、虚警和正负降水偏差的多任务学习，并保存模型 checkpoint 和训练指标。结果保存到：

```text
result/checkpoints/unetdif.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。论文未提供可确认的官方 checkpoint，虚拟数据训练生成的工程 checkpoint 仅用于流程验证，不声明兼容外部权重。

### 推理

```bash
python scripts/inference.py
```

推理加载 checkpoint，按 `bias = ECMWF - CMPA` 应用四个输出头进行非负降水订正。结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估生成逐 3 小时时段和 24 小时累计降水的分类与空间技巧指标，并保存结构化结果。评估同时生成 ECMWF、CMPA、订正降水和误差对比图。虚拟数据结果仅用于验证工程流程，不代表论文正式性能。

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

本仓库为 UnetDif 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
