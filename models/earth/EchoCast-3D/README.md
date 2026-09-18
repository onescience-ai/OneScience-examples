<p align="center"><strong><span style="font-size: 30px;">EchoCast-3D</span></strong></p>

# 模型介绍

EchoCast-3D 根据三帧三维雷达体扫生成未来五帧概率雷达回波，用于强对流三维演变、缺测重建和短时临近预报。模型在扩散生成过程中联合学习时空演变和垂直结构，在输入回波缺失时仍可形成完整 ensemble 预测。

论文：Generative machine learning for skilful 3D radar nowcasting  
https://doi.org/10.1038/s41612-026-01407-7

# 模型描述

该方法由中国科学院、河海大学等机构的研究团队提出。论文使用中国气象局四仰角雷达体扫资料训练和评估模型。EchoCast-3D 将多仰角楔形雷达块编码为统一 token，并通过 MaskDiT 扩散和遮盖重建联合学习回波演变与缺测恢复。模型适用于过去 18 分钟到未来 30 分钟的三维概率雷达临近预报。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 三维雷达临近预报 | 根据 3 帧历史雷达预测未来 5 帧回波。 |
| 缺测鲁棒预测 | 联合执行 75% token masking 重建与扩散去噪。 |
| 本地工程验证 | 在真实 packed wedge 几何上验证 ensemble 预报和指标。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证数据、训练、推理、雷达指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/EchoCast-3D --local_dir ./EchoCast-3D
cd EchoCast-3D
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

本仓库使用少量虚拟样本验证工程流程，虚拟数据包含 8 个连续 6 分钟雷达体扫、4 个仰角、3 帧历史、5 帧目标和真实 packed-wedge 尺度。四个仰角保持 `366/366/363/363` 个方位和 `180/180/120/120` 个距离库，仅减少样本数量、模型规模、扩散步数和训练轮次；按 `3×3` patch 计算为 24,320 个 token，而论文报告 24,400 个。该数据仅用于验证 MaskDiT、缺测重建、扩散训练、ensemble 推理和评估流程，不代表中国气象局官方雷达数据分布与训练规模。

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

训练对未遮盖 token 计算 score matching loss，并对 75% 随机遮盖 token 增加 `0.1` 倍重建损失。默认配置缩小 hidden dimension、DiT 深度、注意力头数、扩散步数、样本和训练轮数，保留雷达几何及 3→5 帧协议。训练产物保存到：

```text
result/checkpoints/echocast_3d.pt
result/training/metrics.json
```

### 训练权重

论文未给出可确认的公开预训练权重链接，本仓库不在 `weight/` 中内置权重。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，以过去三帧雷达体扫和对应有效掩码作为条件。模型从三维高斯噪声开始迭代去噪，生成未来五帧雷达回波，并通过不同随机种子形成 ensemble。输出保持逐时效、逐仰角、逐方位和逐距离顺序。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算 ensemble CRPS、MAE 和 RMSE，并在 20、30 和 40 dBZ 阈值下计算 CSI、FAR 和 POD。结果同时保存五个时效、四个仰角的 ensemble coverage ratio，并生成逐时效观测与 ensemble mean 的复合反射率对比图。虚拟数据结果仅用于工程验证，不代表论文正式性能。评估结果保存到：

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

本仓库为 EchoCast-3D 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY-NC-ND 4.0 许可证；论文、官方模型权重和中国气象局雷达数据仍应按照各自项目的许可证及使用条款使用。
