<p align="center">
  <strong><span style="font-size: 30px;">ClimateSet-Emulator</span></strong>
</p>

# 模型介绍

ClimateSet-Emulator 根据连续 12 个月的 CO2、CH4、SO2 和 BC 人为强迫场模拟同期近地表气温与降水场，为 ClimateSet 气候模式模拟任务提供可运行的工程验证流程。

论文：ClimateSet: A Large-Scale Climate Model Dataset for Machine Learning  
https://arxiv.org/abs/2311.03721

# 模型描述

该方法由 Mila Quebec AI Institute、McGill University 和 University of Osnabrück 等机构的研究团队提出。
论文使用来自 Input4MIPs 和 CMIP6 档案的 36 个气候模式输入与输出数据开展气候模式模拟基准实验。
模型适用于根据人为强迫场模拟月尺度近地表气温和降水场，并评估不同排放情景间的泛化能力。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 月尺度气候场模拟 | 根据连续 12 个月的 CO2、CH4、SO2 和 BC 场模拟同期 `tas` 与 `pr` 场。 |
| 留一情景评估 | 使用 historical、SSP1-2.6、SSP3-7.0 和 SSP5-8.5 训练，并在独立 SSP2-4.5 情景上评估泛化能力。 |
| 本地工程验证 | 使用结构化虚拟数据验证数据生成、训练、checkpoint 恢复、推理和逐变量逐月评估流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理和逐变量逐月气候指标流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/ClimateSet-Emulator --local_dir ./ClimateSet-Emulator
cd ClimateSet-Emulator
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

论文使用 Input4MIPs 人为强迫输入和 CMIP6 气候模式输出，覆盖 36 个气候模式、5 种情景、4 种强迫因子以及近地表气温和降水变量。
本仓库数据契约为输入 `X[B,12,4,96,144]` 和目标 `Y[B,12,2,96,144]`，默认生成 4 个训练样本与 2 个测试样本，并保留真实 `96×144` 空间网格。
虚拟数据仅用于验证工程流程，不代表 Input4MIPs 或 CMIP6 的真实数据分布、训练规模或论文性能。

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

训练使用纬度加权均方误差，默认配置采用 4 个全分辨率虚拟训练样本、批次大小 1 和 1 个训练轮次。
正式实验需要真实 ClimateSet 数据与完整计算资源，训练产物保存到：

```text
result/checkpoints/climateset_emulator.pt
result/training/metrics.json
```

### 训练权重

ClimateSet 官方 checkpoint 的完整链接为 https://huggingface.co/climateset/causalpaca_models 。本仓库不在 `weight/` 中内置官方权重；本地训练生成的 `result/checkpoints/climateset_emulator.pt` 是当前工程实现与数据对应的 checkpoint，不声明与官方 checkpoint 直接兼容。

### 推理

```bash
python scripts/inference.py
```

推理加载本地训练 checkpoint，在 SSP2-4.5 测试数据上生成 12 个月的 `tas` 和 `pr` 场，完整数值结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估按 `tas`、`pr` 和月份计算 SSP2-4.5 情景上的纬度加权 RMSE；当前脚本不生成可视化图片，指标保存到：

```text
result/evaluation/metrics.json
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 ClimateSet-Emulator 公开规格的独立工程复现版本。

本仓库代码采用 Apache-2.0 许可证；ClimateSet 官方代码采用 GPL-3.0，官方模型权重、数据和其他第三方资源的使用仍应以各自项目中的许可证及使用条款为准。
