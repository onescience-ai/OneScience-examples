<p align="center">
  <strong>
    <span style="font-size: 30px;">DGMR</span>
  </strong>
</p>


# 模型介绍

DGMR（Deep Generative Model of Radar，雷达回波深度生成模型）由 DeepMind 于 2021 年提出，是基于条件生成对抗网络（cGAN）的短临降水临近预报模型。生成器由潜变量条件栈、上下文条件栈和自回归采样器（多层 ConvGRU）构成，判别器同时对空间与时间维度进行判定，训练目标为 hinge GAN 损失加网格单元正则器。

论文：Skillful Precipitation Nowcasting using Deep Generative Models of Radar

https://arxiv.org/abs/2104.00954

# 模型描述

DGMR 是概率性的短临降水预报模型：输入连续 4 帧雷达回波场，一次生成未来 18 帧（论文中步长 5 分钟、共 90 分钟）的雷达回波场，输出为样本而非确定性估计。本仓库基于 Open Climate Fix 社区官方 PyTorch 实现（`openclimatefix/dgmr`，MIT License）整理，并接入 OneScience 数据读取与训练流程。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 短临降水预报研究 | 基于雷达回波序列训练 cGAN 生成未来若干帧回波场。 |
| 概率性预报输出 | 通过潜变量采样获得未来场的多个生成样本。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、训练入口、推理和结果脚本。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。


### 下载模型包

```bash
modelscope download --model OneScience/DGMR --local_dir ./DGMR
cd DGMR
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

OneScience 社区提供可供训练的 ERA5 数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认 `conf/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

如需快速验证流程，可先运行虚拟数据脚本：

```bash
python scripts/fake_data.py
```

> 注：`scripts/fake_data.py` 根据 `num_context`、`forecast_steps`、batch size 和雷达网格尺寸生成单通道雷达序列；当前小配置对应 4 帧输入、6 帧输出、128×128 网格。

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
data/checkpoints/model_bak.pth
data/checkpoints/trloss.npy
data/checkpoints/valoss.npy
```

### 训练权重
本仓库在 weight/ 文件夹内预留权重目录，默认不提供预训练权重，用户可依据论文配置自行训练。官方（DeepMind）预训练权重需申请权限获取，且与当前配置的通道数/网格大小不一致，加载前需自行对齐。

### 推理

推理默认读取 `data/checkpoints/model_bak.pth`：

```bash
python scripts/inference.py
```

预测结果按预报帧时刻逐帧输出到：

```text
result/output/
```

### 评估与可视化

```bash
python scripts/result.py
```

输出内容包括：

- `result/rmse.npy`
- `result/acc.npy`
- `result/loss.png`
- 指定日期和变量的预报对比图


# 官方来源与复现说明

- 模型实现来自 Open Climate Fix 社区官方 PyTorch 包 `openclimatefix/dgmr`（MIT License），相关网络模块（common/layers/generators/discriminators/losses）以 `model/dgmr_official/` 原样内嵌（剥离 HuggingFace hub mixin 与 pytorch_lightning 训练循环），`model/dgmr.py` 仅为 YAML 驱动的薄包装。
- `conf/config.yaml` 默认使用小配置（`forecast_steps=6`、`output_shape=128`、`latent_channels=384`、`context_channels=192`）用于连通性验证；论文级复现需调整为 4→18 帧、256×256、`latent_channels=768`、`context_channels=384`。
- 论文级配置（`num_context=4`、`forecast_steps=18`）要求数据每年帧数 `T >= num_context + forecast_steps + 1 = 23`；当前虚拟数据 `T=10`（time_step=6h），故默认配置取 4→6 帧。
- 判别器内部含 BatchNorm1d，训练 batch 必须 `>= 2`（虚拟数据下训练集取两年共 2 个样本，恰好 1 个 batch）。
- 论文中以下细节论文未公开，复现时为假设项：数据归一化统计量（当前使用恒等归一化，真实统计量随数据接入）、判别器随机采样的时间帧数等。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 DGMR 的独立工程整理与适配，模型源码参考自 Open Climate Fix 的 `openclimatefix/dgmr` 实现，遵循 MIT License。
- 引用请参考：Ravuri et al. Skilful Precipitation Nowcasting using Deep Generative Models of Radar. Nature 597, 672-677, 2021.

