<p align="center"><strong><span style="font-size: 30px;">TropiCycloneNet</span></strong></p>

# 模型介绍

TropiCycloneNet 是同时预测全球热带气旋轨迹和强度的多模态深度学习方法。模型融合气旋固有属性、局地气象场与环境信息，并生成多种可能的发展路径。

论文：Benchmark dataset and deep learning method for global tropical cyclone forecasting  
https://doi.org/10.1038/s41467-025-61087-4

# 模型描述

该模型由浙江工业大学、山东大学、天津理工大学和浙江省视觉信息智能处理重点实验室的研究团队提出。模型使用 TCND 中覆盖六大海域、近 70 年的气旋最佳路径、ERA5 气象场和环境特征训练。模型通过联合时序、空间和环境编码以及多个生成器，适用于未来 24 小时全球热带气旋轨迹、中心气压和最大持续风速预报。

# 适用场景

| 场景 | 说明 |
|---|---|
| 气旋轨迹预报 | 预测未来 6–24 小时经纬度。 |
| 气旋强度预报 | 预测中心气压和最大持续风速。 |
| 多模态建模 | 融合 Data1d、Data3d 和 Env-Data。 |
| 多趋势预报 | 输出多个生成器候选路径。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、概率降水指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/TropiCycloneNet --local_dir ./TropiCycloneNet
cd TropiCycloneNet
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

论文使用 8 个六小时时刻的 4 个气旋属性、中心附近 `81×81` 的 500 hPa 位势高度和环境变量，预测后续 4 个时刻。虚拟数据保持各模态 shape、时间顺序、六海域协议和 6 个生成器，仅减少气旋样本、隐藏宽度和训练轮数。结果仅用于工程验证，不代表论文性能。

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
torchrun --nproc_per_node=2 --nnodes=1 --master_addr="localhost" --master_port=29500 scripts/train.py
```

默认虚拟数据训练能够完成多模态前向传播、反向传播和参数更新，单卡与双进程 DDP 流程均已验证通过。训练完成后生成包含模型参数与配置的单一 checkpoint，并记录训练损失。训练结果保存到：

```text
result/checkpoints/tropicyclonenet.pt
result/training/metrics.json
```

### 训练权重

论文公开代码与资源位于 https://github.com/xiaochengfuhuo/TropiCycloneNet ，本仓库未确认可直接加载且许可证明确的官方预训练权重，因此不提供权重链接。

### 推理

```bash
python scripts/inference.py
```

推理恢复 checkpoint，输出 6 个生成器在 6、12、18、24 小时的候选轨迹和强度，并保存生成器概率。默认虚拟样本的输出维度和有限数值检查均已通过。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算轨迹大圆距离 MAE、气压 MAE 和风速 MAE，并绘制逐时效误差与路径对照。评估指标均为有限数值，生成的对比图已通过格式和有效像素检查；虚拟结果不代表论文指标。评估结果保存到：

```text
result/evaluation/metrics.json
result/evaluation/comparison.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 TropiCycloneNet 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY 4.0 许可证；论文、TCND 数据、官方代码和权重仍应按照各自项目的许可证及使用条款使用。
