<p align="center">
  <strong>
    <span style="font-size: 30px;">CorrDiff</span>
  </strong>
</p>

# 模型介绍

CorrDiff 是用于公里尺度大气降尺度的两阶段生成模型，先通过条件回归预测高分辨率均值，再使用残差扩散模型生成局地随机细节和集合预报。

论文：Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling  
https://arxiv.org/abs/2309.15214

# 模型描述

CorrDiff 由 NVIDIA 研究团队提出。模型使用 ERA5 再分析资料与 CWA-WRF 高分辨率区域模拟数据训练，通过回归和残差扩散完成气象场降尺度。模型适用于高分辨率气象场生成、公里尺度降尺度和集合不确定性分析。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 气象降尺度 | 将 `36x36` 粗分辨率条件场降尺度到 `448x448`。 |
| 集合预报 | 通过扩散采样生成多个可能的高分辨率结果。 |
| 雷达反射率生成 | 根据 ERA5 条件场生成输入中不存在的高分辨率最大雷达反射率，支持降水系统精细结构分析。 |
| 极端天气风险分析 | 利用集合成员和空间不确定性刻画局地强降水等高影响天气的可能演变。 |
| 本地工程验证 | 使用少量虚拟数据检查训练、推理和评估流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/CorrDiff --local_dir ./CorrDiff
cd CorrDiff
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

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

默认使用 2 个同协议虚拟样本验证工程流程，数据保存为 `data/corrdiff.npz`，虚拟数据不代表真实气象分布。

虚拟数据保持论文台湾降尺度实验的 12 通道 `36x36` 条件场和 4 通道 `448x448` 目标场输入输出规格。

真实数据需预处理并转换为以下 NPZ 训练协议；该协议与模型输入规格一致，但不等同于原始数据集的下载格式。

```text
input: float32 [N,12,36,36]
target: float32 [N,4,448,448]
```

`fake_data.py` 会自动写入 `protocol` 和 `data_source` 协议元数据，使用真实数据时需保留这些字段。

```bash
python scripts/fake_data.py
```

### 训练

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 scripts/train.py
```

训练依次优化条件回归和残差扩散阶段，并保存 checkpoint 与总体训练指标。默认配置面向快速流程验证；开展正式实验时，应使用论文对应的数据规模、模型配置和训练周期。

```text
result/checkpoints/corrdiff.pt
result/training/metrics.json
```

### 训练权重

本仓库将在 `weight/` 文件夹内提供 CorrDiff 训练权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，按配置生成高分辨率集合预报，并将结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估总体报告 MAE、RMSE、集合 CRPS 和集合离散度，并生成集合诊断图。虚拟数据结果仅用于验证工程流程，不代表论文完整性能。

```text
result/evaluation/metrics.json
result/evaluation/ensemble_diagnostics.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 CorrDiff 原始论文的复现版本。

本仓库代码和数据的使用仍应以各自项目中的许可证及使用条款为准。
