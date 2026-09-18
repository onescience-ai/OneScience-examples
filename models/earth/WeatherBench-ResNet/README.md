<p align="center">
  <strong><span style="font-size: 30px;">WeatherBench-ResNet</span></strong>
</p>

# 模型介绍

WeatherBench-ResNet 通过残差神经网络从历史大气状态直接预测多个未来时效的全球天气场，为数据驱动的中期天气预报研究提供可运行的工程验证流程。

论文：Data-Driven Medium-Range Weather Prediction With a Resnet Pretrained on Climate Simulations  
https://doi.org/10.1029/2020MS002405

# 模型描述

该方法由慕尼黑工业大学（TUM）的作者团队提出。
论文先使用 CMIP 中 MPI-ESM-HR 气候模拟数据预训练模型，再使用 ERA5 再分析数据微调。
模型执行 Direct 固定时效和 Continuous 时效条件化两类中期天气预报任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气要素预测 | 在 `32×64` 全球网格上预测 Z500、T850 和 T2M。 |
| Direct/Continuous 任务验证 | 验证固定预报时效的 Direct 任务和由预报时效条件控制的 Continuous 任务。 |
| 工程流程验证 | 使用结构化虚拟数据验证 CMIP 预训练、ERA5 微调、非自回归推理和逐时效评估流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证数据生成、训练、推理和天气预报指标流程。 |
| 多卡训练 | 通过标准 `torchrun` 命令验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/WeatherBench-ResNet --local_dir ./WeatherBench-ResNet
cd WeatherBench-ResNet
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的流程验证。
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

论文使用 CMIP 中 MPI-ESM-HR 气候模拟数据进行预训练，并使用 ERA5 再分析数据微调，当前工程在 `32×64` 全球网格上组织历史场和目标场。
仓库实现明确采用 Direct 的 117 通道边界（3 个时刻乘 38 个动态通道，再加 3 个静态通道）和 Continuous 的 118 通道边界（额外加入 1 个预报时效通道），同时保留论文所述 114 通道与该输入定义之间的冲突记录。
默认生成 8 个结构化虚拟样本，仅用于验证数据契约和工程流程，不代表真实 CMIP、MPI-ESM-HR 或 ERA5 数据分布、训练规模及论文性能。

```bash
python scripts/fake_data.py --force
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

训练依次执行 CMIP 预训练和 ERA5 微调，并使用纬度加权均方误差优化天气要素预测。
默认配置用于小样本工程验证，正式实验需要真实数据和完整计算资源，训练产物保存到：

```text
result/checkpoints/weatherbench_resnet.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置官方权重，也未发现论文作者发布的可确认、可直接加载的官方预训练 checkpoint。本地训练生成的 `result/checkpoints/weatherbench_resnet.pt` 是当前数据对应的工程 checkpoint，不得描述为官方权重或声明与官方权重兼容。

### 推理

```bash
python scripts/inference.py
```

推理加载完成 ERA5 微调的本地 checkpoint，对每个预报时效复用同一组 3 帧历史观测，不将预测结果反馈为后续输入。数值结果及元数据保存到：

```text
result/output/predictions.npz
result/output/predictions.metadata.json
```

### 评估和可视化

```bash
python scripts/result.py
```

评估对 Z500、T850 和 T2M 逐时效计算纬度加权 RMSE 与 ACC，结果保存到 `result/evaluation/metrics.json`。当前 `scripts/result.py` 不生成可视化图片，因此本节仅列出数值指标产物；虚拟数据指标只用于验证工程流程，不代表论文正式性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 WeatherBench-ResNet 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
