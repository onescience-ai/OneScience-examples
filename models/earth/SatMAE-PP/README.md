<p align="center">
  <strong>
    <span style="font-size: 30px;">SatMAE++</span>
  </strong>
</p>

# 模型介绍

SatMAE++ 是面向光学和多光谱卫星影像的掩码自编码器，使用可见 token 编码器学习遥感表征，并通过卷积多尺度 decoder 重建原生空间尺度目标。模型支持 RGB 与分组 Sentinel 输入，训练目标由掩码 patch 的 MSE+L1 和多尺度重建的 MSE+L1 组成。

论文：Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery  
https://arxiv.org/abs/2403.05419

# 模型描述

SatMAE++ 由阿布扎比穆罕默德·本·扎耶德人工智能大学的 Mubashir Noman、Muzammal Naseer、Hisham Cholakkal、Rao Muhammad Anwar、Salman Khan 和 Fahad Shahbaz Khan 团队提出。模型使用 FMoW-RGB 和 FMoW-Sentinel 影像训练。模型适用于光学卫星影像与多光谱卫星影像的表征学习和多尺度重建任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 光学卫星影像预训练 | RGB 输入，使用 1x 和 2x 原生目标。 |
| 多光谱卫星影像预训练 | 分组 Sentinel 输入，使用 1x、2x 和 4x 原生目标。 |
| 遥感场景分类 | 迁移模型表征并微调，用于 EuroSAT、UCMerced 和 RESISC-45 等场景分类数据集。 |
| 多标签地表覆盖分类 | 在 BigEarthNet 等多光谱数据集上微调，识别同一区域内的多种地表覆盖类型。 |
| 本地快速验证 | 使用虚拟数据完成训练、推理、评估和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动 DDP 训练。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于小配置连通性验证。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/SatMAE++ --local_dir ./SatMAE++
cd SatMAE++
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

论文使用 FMoW-RGB 和 FMoW-Sentinel 进行预训练。NPZ 文件中的 `images` 为 `[B,C,H,W]`，并提供与配置尺度一致的 `images_2x` 或 `images_4x` 原生目标。RGB 使用三通道，Sentinel 使用十通道并按光谱组编码。

默认使用虚拟数据：

```bash
python scripts/fake_data.py
```

使用虚拟数据时运行上面的命令。使用真实数据时不要运行 `fake_data.py`，请将数据整理为 `data/train.npz` 和 `data/test.npz`，其中 RGB 数据至少包含：

```text
images:    float32 [N,3,H,W]
images_2x: float32 [N,3,2H,2W]
```

Sentinel 数据使用十个通道，并根据配置提供 `images_2x` 和 `images_4x`：

```text
images:    float32 [N,10,H,W]
images_2x: float32 [N,10,2H,2W]
images_4x: float32 [N,10,4H,4W]
```

高分辨率字段应是与输入场景配准的原生目标，不要用低分辨率影像临时插值替代。准备好数据后，根据实际通道数、输入尺寸、光谱分组和训练尺度修改 `conf/config.yaml`。

### 训练

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 scripts/train.py
```

训练输出：

```text
result/checkpoints/satmae_pp.pt
result/training/metrics.json
```

训练输出包括可用于后续推理的模型检查点，以及反映训练过程和损失收敛情况的指标记录，便于保存训练状态并分析模型优化效果。

默认配置是快速虚拟数据协议。论文正式 RGB 配置为 ViT-L、`224/16`、800 epochs；Sentinel 配置为 ViT-L、`96/8`、去除 B1/B9/B10 后的 10 通道三组输入、50 epochs。

### 训练权重

本仓库在 `weight/` 文件夹内提供基于 FMoW-RGB 光学卫星影像和 FMoW-Sentinel 多光谱卫星影像训练的权重，权重文件即将上传，预计将于近期完成。

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
result/evaluation/multiscale_reconstruction.png
result/evaluation/scale_comparison.png
```

评估结果从整体重建误差、掩码区域恢复质量和不同原生尺度的重建误差等方面衡量模型表现，并通过多尺度重建效果及尺度间误差对比直观展示模型的多尺度恢复能力。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 SatMAE++ 原始论文的复现版本。
