<p align="center">
  <strong>
    <span style="font-size: 30px;">RemoteCLIP</span>
  </strong>
</p>

# 模型介绍

RemoteCLIP 是面向遥感影像与文本的视觉语言基础模型，通过 CLIP 双编码器和双向对比学习对齐遥感视觉语义与自然语言描述，可支持跨模态检索和遥感下游任务迁移。

论文：RemoteCLIP: A Vision Language Foundation Model for Remote Sensing  
https://arxiv.org/abs/2306.11029

# 模型描述

RemoteCLIP 由国防科技大学等机构的研究团队提出。模型使用 RSITMD、RSICD、UCM-Captions 及经任务数据转换构建的遥感图文数据进行持续预训练。模型适用于遥感图文检索、零样本分类和视觉语言表征学习。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 遥感图文检索 | 计算影像与文本特征的跨模态相似度。 |
| 多正样本对比学习 | 通过 `pair_ids` 表达一图多文等正样本关系。 |
| 零样本场景分类 | 使用自然语言类别提示与影像特征匹配，在不额外训练分类头的情况下识别遥感场景。 |
| 少样本视觉识别 | 迁移视觉语言表征并进行少样本微调或线性探测，适配标注有限的遥感分类任务。 |
| 本地工程验证 | 使用少量虚拟数据检查训练、推理和评估流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/RemoteCLIP --local_dir ./RemoteCLIP
cd RemoteCLIP
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

默认使用 8 个训练和 4 个测试虚拟图文样本验证工程流程，分别保存为 `data/train.npz` 和 `data/test.npz`。Token 遵循 OpenAI CLIP BPE 的 49408 词表以及 SOT、EOT、padding 序列约束。

虚拟数据保持作者官方模型的 3 通道 `224x224` 图像和长度 77 的 CLIP 文本序列输入规格。

真实数据需预处理并转换为以下 NPZ 训练协议；该协议与模型输入规格一致，但不等同于原始数据集的下载格式。

```text
images: float32 [N,3,224,224]
tokens: int64 [N,77]
pair_ids: int64 [N]
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

训练优化图像和文本双编码器的多正样本双向对比目标，并保存 checkpoint 与总体训练指标。默认配置面向快速流程验证；开展正式实验时，应使用论文对应的图文数据规模、模型配置和训练周期。

```text
result/checkpoints/remoteclip.pt
result/training/metrics.json
```

### 训练权重

本仓库将在 `weight/` 文件夹内提供 RemoteCLIP 训练权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，计算测试集图像与文本特征及相似度，并将结果保存到：

```text
result/output/retrieval.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估依据 `pair_ids` 总体报告双向检索的 R@1、R@5、R@10 和平均召回率，并生成相似度热图。虚拟数据结果仅用于验证工程流程，不代表论文完整性能。

```text
result/evaluation/metrics.json
result/evaluation/similarity_matrix.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 RemoteCLIP 原始论文的复现版本。

本仓库代码和数据的使用仍应以各自项目中的许可证及使用条款为准。
