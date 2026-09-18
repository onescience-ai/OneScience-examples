<p align="center">
  <strong>
    <span style="font-size: 30px;">SkySense</span>
  </strong>
</p>

# 模型介绍

SkySense 是面向通用地球观测解译的多模态遥感基础模型，通过独立空间编码、时序聚合、跨模态融合和地理原型建模联合表示高分辨率 RGB、Sentinel-1 与 Sentinel-2 数据。

论文：SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery  
https://arxiv.org/abs/2312.10115

# 模型描述

SkySense 由武汉大学、华中科技大学等机构的研究团队提出。模型使用大规模多模态遥感时序数据训练，融合高分辨率 RGB、Sentinel-1 雷达和 Sentinel-2 多光谱观测。模型适用于多模态遥感表征学习、土地覆盖语义分割及其他地球观测解译任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 多模态遥感融合 | 融合 HR、Sentinel-1 和 Sentinel-2 的多分辨率观测。 |
| 多时序建模 | 使用日期编码和模态内时间聚合处理卫星序列。 |
| 语义分割 | 输出高分辨率土地覆盖类别图。 |
| 地物目标检测 | 迁移多模态表征并微调，用于飞机、船舶和车辆等遥感目标检测。 |
| 地表变化检测 | 对多时相观测进行下游适配，识别建筑物和土地覆盖变化区域。 |
| 本地工程验证 | 使用少量虚拟数据检查训练、推理和评估流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/SkySense --local_dir ./SkySense
cd SkySense
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

默认使用 2 个训练和 1 个测试虚拟样本验证工程流程，分别保存为 `data/train.npz` 和 `data/test.npz`。

虚拟数据保持论文及作者官方骨干的静态 HR、20 时相 Sentinel-2、10 时相 Sentinel-1，以及各模态通道数和空间尺寸规格。

真实数据需预处理并转换为以下 NPZ 训练协议；该协议与模型输入规格一致，但不等同于原始数据集的下载格式。

```text
hr: float32 [N,1,3,224,224]
s2: float32 [N,20,10,64,64]
s1: float32 [N,10,2,64,64]
dates_hr: int64 [N,1]
dates_s2: int64 [N,20]
dates_s1: int64 [N,10]
region: int64 [N]
labels: int64 [N,224,224]
band_order_hr: string [3]
band_order_s2: string [10]
band_order_s1: string [2]
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

训练联合优化语义分割和跨模态表征对齐目标，并保存 checkpoint 与总体训练指标。默认配置面向快速流程验证；开展正式实验时，应使用论文对应的多模态时序数据、模型配置和训练周期。

```text
result/checkpoints/skysense.pt
result/training/metrics.json
```

### 训练权重

本仓库将在 `weight/` 文件夹内提供 SkySense 训练权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，按批次生成高分辨率语义分割结果，并保存到：

```text
result/output/
```

### 评估和可视化

```bash
python scripts/result.py
```

评估总体报告像素准确率、各类别 IoU 和平均 IoU，并生成输入、标签与预测对比图。虚拟数据结果仅用于验证工程流程，不代表论文完整性能。

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

本仓库为 SkySense 原始论文的复现版本。

本仓库代码和数据的使用仍应以各自项目中的许可证及使用条款为准。
