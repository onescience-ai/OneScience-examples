<p align="center">
  <strong><span style="font-size: 30px;">MetNet-2</span></strong>
</p>

# 模型介绍

MetNet-2 根据雷达、卫星和大气状态生成未来 12 小时的高分辨率概率降水预报，主要用于短时降水预测和极端降水风险分析。

论文：Deep learning for twelve hour precipitation forecasts  
https://doi.org/10.1038/s41467-022-32483-x

# 模型描述

该方法由 Google Research 团队提出。论文使用 MRMS 雷达、GOES 卫星和 HRRR 同化大气状态构建 2017-2020 年训练与测试数据。模型适用于以 2 分钟间隔预测未来 12 小时的逐网格降水概率分布。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 降水分类预报 | 验证 512 类条件降水分布及 12 小时时效协议。 |
| 核心方法验证 | 验证 ConvLSTM、lead-time FiLM 和多尺度膨胀残差栈。 |
| 本地工程验证 | 使用确定性 procedural field 验证完整 `641×512×512` 逻辑协议，不物化完整输入。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、概率降水指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/MetNet-2 --local_dir ./MetNet-2
cd MetNet-2
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

论文训练数据由 MRMS、HRRR、GOES、静态地理和时间信息组成，并遵循 641 通道、`512×512` 空间域协议。本仓库生成 8 条确定性窗口记录，运行时仅构造选定的 `32×32` 窗口及 halo，同时保留完整逻辑形状和通道分组。合成数据只验证工程连通性，不代表真实气象数据分布、论文训练规模或论文性能。

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

默认配置仅缩小样本数、网络宽度、残差块数量和训练步数，不缩小 641 通道、512 个降水类别和 12 小时时效协议。正式实验需要真实 MRMS、GOES 和 HRRR 数据及完整计算资源，训练产物保存到：

```text
result/checkpoints/metnet_2.pt
result/training/metrics.json
```

### 训练权重

论文未提供可直接加载的官方模型权重，本仓库不在 `weight/` 中内置权重。本地训练生成的 checkpoint 保存到 `result/checkpoints/metnet_2.pt`，不得描述为官方预训练权重。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，为选定空间窗口生成 512 类降水概率及对应 CDF，并明确保存覆盖范围和完整性标记。完整数值结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算离散 CRPS、Brier Score 和不同降水阈值下的 CSI，并生成目标、期望降水率和误差对比图。虚拟数据结果仅用于验证工程流程，不代表论文正式性能；结果保存到：

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

本仓库为 MetNet-2 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
