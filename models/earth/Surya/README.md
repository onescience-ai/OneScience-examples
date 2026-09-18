<p align="center">
  <strong><span style="font-size: 30px;">Surya</span></strong>
</p>

# 模型介绍

Surya 是面向太阳物理和空间天气研究的时空基础模型，通过多仪器 SDO 观测学习太阳动力学。模型融合谱门控与长短程注意力，以两个历史时刻预测未来太阳图像，并支持自回归多步预报。

论文：Surya: Foundation Model for Heliophysics  
https://ntrs.nasa.gov/citations/20250008498

# 模型描述

Surya 由 NASA、IBM Research 和 University of Alabama in Huntsville 等机构提出，使用 SDO 的八个 AIA 通道与五个 HMI 产品作为训练数据。模型适用于太阳图像的一步预测和自回归多步预报任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 太阳动力学预报 | 根据两个历史时刻预测未来多通道太阳图像。 |
| 空间天气研究 | 分析太阳活动和不同预报时效下的误差。 |
| SDO 多仪器建模 | 联合处理 AIA 与 HMI 多通道观测。 |
| 太阳耀斑预测 | 微调模型表征，预测未来时段内 M 级和 X 级太阳耀斑。 |
| 太阳活动区分割 | 基于磁图微调，分割太阳活动区和磁极性反转线。 |
| 本地快速验证 | 使用虚拟数据验证训练、推理、评估和可视化。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于当前默认小配置的流程验证。
- 原论文 4096×4096 分辨率、3.66 亿参数训练需要大规模多卡计算资源。

### 下载模型包

```bash
modelscope download --model OneScience/Surya --local_dir ./Surya
cd Surya
```

### 安装运行环境

**DCU环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

论文使用 2010 至 2024 年的 SDO 数据，包括 AIA 94、131、171、193、211、304、335、1600 Å，以及 HMI 的磁场与多普勒速度产品。数据统一到 12 分钟时间间隔并进行空间配准。原始训练数据约 257 TB。

本模型包默认使用虚拟数据：

```bash
python scripts/fake_data.py
```

每个 NPZ 文件至少包含 `inputs`、`targets` 和 `activity`。`inputs` 采用 `[N, 2, 13, H, W]`，`targets` 采用 `[N, S, 13, H, W]`，`activity` 保存各预报时刻的太阳活动强度。使用真实数据时按相同字段准备文件并修改 `conf/config.yaml`。

使用真实数据时，不要运行 `fake_data.py`。请将经过时间对齐、空间配准、通道整理和 signum-log 归一化的数据保存到 `data/`，替换虚拟数据文件：

```text
data/train.npz
data/test.npz
```

每个 NPZ 文件至少包含：

```text
inputs:   float32 [N,2,13,H,W]
targets:  float32 [N,S,13,H,W]
activity: float32 [N,S]
```

其中 13 个通道依次对应 8 个 AIA 通道和 5 个 HMI 产品，`S` 与 `conf/config.yaml` 中的 `forecast_steps` 一致。根据真实数据的图像尺寸、预报步数、通道统计量和数据路径修改 `conf/config.yaml`。完成数据准备后，继续使用下面统一的训练、推理和评估命令；如需使用其他文件位置，再通过脚本参数覆盖默认路径。

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练输出：

```text
result/checkpoints/surya.pt
result/training/metrics.json
```

训练输出包括可用于后续多步预报的模型检查点，以及反映不同训练阶段、整体损失和学习率变化的训练指标，便于保存训练状态并分析模型优化效果。虚拟数据结果仅用于验证代码流程，不代表论文结果。

### 训练权重

本仓库在 `weight/` 文件夹内提供基于 SDO/AIA 与 SDO/HMI 太阳观测数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理结果输出到：

```text
result/output/forecast.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估和可视化输出到：

```text
result/evaluation/metrics.json
result/evaluation/rollout_forecast_skill.png
result/evaluation/solar_dynamics_forecast.png
result/evaluation/solar_activity_evolution.png
result/evaluation/sdo_channel_error.png
result/evaluation/persistence_skill.png
```

评估结果综合衡量多步预报误差、相对持久性基线的预报技巧、AIA/HMI 各通道表现和太阳活动强度变化，并通过预报图像与时效曲线展示太阳动力学的演化和模型的长期预报稳定性。结果对应论文的自回归预报协议；虚拟数据结果只用于验证 Surya 流程，不代表论文下游任务性能。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 Surya 原始论文的复现版本。
