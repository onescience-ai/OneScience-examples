<p align="center">
  <strong>
    <span style="font-size: 30px;">AlphaEarthFoundations</span>
  </strong>
</p>

# 模型介绍

AlphaEarthFoundations 是面向全球地球观测制图的多源时空嵌入场模型，将光学、雷达、LiDAR、气候、高程、土地覆盖和地理文本等稀疏且异步的数据统一编码为 64 维单位球嵌入，以少量标签支持分类、回归和变化检测。

论文：AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data  
https://arxiv.org/abs/2507.22291

# 模型描述

AlphaEarth Foundations 由 Google DeepMind 与 Google 的研究团队提出。模型使用超过 30 亿次观测训练，输入 Sentinel-2、Sentinel-1 和 Landsat-8/9 时序影像，并以 PALSAR-2、ERA5-Land、GEDI、GRACE、Copernicus DEM、NLCD 和地理文本等数据作为学习目标。模型适用于多源地球观测表征学习、稀疏标签专题制图、生物物理变量估计和时序变化检测。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 多源时空表征 | 融合 Sentinel-2、Sentinel-1 和 Landsat-8/9 的异步时序观测。 |
| 稀疏标签制图 | 使用 64 维 embedding 和少量点标签训练 kNN 或线性预测器。 |
| 生物物理变量估计 | 基于 embedding 回归地表发射率、蒸散量等连续变量。 |
| 地表变化检测 | 比较不同有效时段的单位球 embedding，执行监督或无监督变化检测。 |
| 本地工程验证 | 使用保持论文真实维度的少量虚拟数据检查训练、推理、量化、评估和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/AlphaEarthFoundations --local_dir ./AlphaEarthFoundations
cd AlphaEarthFoundations
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证，完整训练和全球推理需要大规模加速资源。
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

本仓库使用少量虚拟样本验证工程流程，训练数据和测试数据分别保存为 `data/train.npz` 和 `data/test.npz`。虚拟数据保持论文的 1.28 km × 1.28 km、10 米网格、Sentinel-2 65 帧、Sentinel-1 17 帧、Landsat 21 帧，以及论文明确公开的输入源和训练目标通道规格；仅样本数量和默认模型内部宽度被缩小。

```bash
python scripts/fake_data.py
```

### 训练

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

默认配置面向快速流程验证；开展正式实验时，应使用论文对应的多时相数据、完整任务标签、模型配置和训练周期。

```text
result/checkpoints/alphaearthfoundations.pt
result/training/metrics.json
```

### 训练权重

本仓库不内置虚拟权重或训练权重。执行训练后生成 `result/checkpoints/alphaearthfoundations.pt`；Google 和 Google DeepMind 尚未公开论文 v2.0/v2.1 模型权重或 checkpoint。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，生成 float32 单位球 embedding、论文 `s8²` 有符号 int8 量化 embedding 和九类目标重建结果，并保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估报告各训练源的重建 MAE 或分类错误率、embedding 平均范数和 `s8²` 量化误差；同时按论文的轻量迁移方式执行 kNN `k=1`、kNN `k=3`、无正则线性分类和线性回归，报告 Balanced Accuracy 与 R²，并生成 A01/A16/A09 三个 embedding 轴与土地覆盖目标的对比图。虚拟数据结果仅用于验证工程流程，不代表论文在 15 个真实下游数据集上的完整性能。

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

本仓库为 AlphaEarth Foundations 原始论文的工程复现版本。

本仓库代码和数据的使用仍应以各自项目中的许可证及使用条款为准。
