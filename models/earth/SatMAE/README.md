<p align="center">
  <strong>
    <span style="font-size: 30px;">SatMAE</span>
  </strong>
</p>

# 模型介绍

SatMAE 是面向时间序列和多光谱卫星图像的掩码自编码器预训练模型，通过时间位置编码、光谱组编码和跨时间或光谱维度的独立遮挡学习遥感图像表示，主要用于在标注数据有限时提升卫星图像分类、土地覆盖分类、多标签分类和语义分割等任务的性能。

论文：SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery  
https://arxiv.org/abs/2207.08051

# 模型描述

SatMAE 由斯坦福大学研究团队提出。模型使用 fMoW 卫星影像、时间序列卫星影像和 Sentinel-2 多光谱影像进行预训练。模型适用于卫星影像分类、土地覆盖分类、多标签分类和语义分割等遥感任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 时间序列卫星图像预训练 | 使用带时间戳的 `BTCHW` 数据训练 SatMAE。 |
| 多光谱卫星图像预训练 | 使用 `BCHW` 数据和光谱分组配置训练 SatMAE。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、训练、推理和评估。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后运行脚本。 |
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
modelscope download --model OneScience/SatMAE --local_dir ./SatMAE
cd SatMAE
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

论文时间序列实验使用 fMoW RGB，输入是同一地点长度为 3 的 RGB 图像序列；多光谱实验使用 fMoW-Sentinel。时间模式 NPZ 中 `images` 为 `[B,T,C,H,W]`，`timestamps` 为 `[B,T,3]`，三个字段依次为 `year - 2002`、`month - 1` 和 `hour`，`labels` 为 `[B]`。模型也接受 `[B,T]` 连续标量时间。

默认使用小尺寸同协议虚拟数据：

```bash
python scripts/fake_data.py
```

使用虚拟数据时运行上面的命令。使用真实数据时不要运行 `fake_data.py`，将数据保存为 `data/train.npz` 和 `data/test.npz`，并根据实际协议修改 `conf/config.yaml`。

```text
images:     float32 [N,T,C,H,W]
timestamps: float32 [N,T,3]
labels:     int64   [N]
```

其中 `timestamps` 的三个字段依次为 `year - 2002`、`month - 1` 和 `hour`；连续标量时间也可使用 `[N,T]` 格式。数据应完成尺寸、通道顺序、时间排序和数值归一化。

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 scripts/train.py
```

训练输出：

```text
result/checkpoints/satmae.pt
result/training/metrics.json
```

默认配置使用小尺寸虚拟数据验证训练流程。正式训练应使用代码提供的 Base、Large 或 Huge 论文结构预设以及真实数据和论文训练预算。

### 训练权重

本仓库在 `weight/` 文件夹内提供基于 fMoW RGB 时间序列卫星影像和 fMoW-Sentinel 多光谱卫星影像训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理结果输出到：

```text
result/output/reconstruction.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估和可视化输出到：

```text
result/evaluation/metrics.json
result/evaluation/temporal_frame_reconstruction.png
result/evaluation/temporal_reconstruction_error.png
result/evaluation/spectral_band_reconstruction.png
```

评估结果包括整体与掩码块 MSE、各时间帧的整体与掩码块 MSE，以及各输入通道的重建 MSE。虚拟数据结果仅用于验证工程流程，不代表论文下游迁移性能。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 SatMAE 原始论文的复现版本。
