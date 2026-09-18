<p align="center">
  <strong>
    <span style="font-size: 30px;">TerraMind</span>
  </strong>
</p>

# 模型介绍

TerraMind 是面向地球观测数据的任意模态到任意模态生成式基础模型，同时处理像素级和离散 Token 级数据，学习雷达、光学、高程、土地覆盖、植被指数、地理坐标和文本之间的跨模态关系。

论文：TerraMind: Large-Scale Generative Multimodality for Earth Observation  
https://arxiv.org/abs/2504.11171

# 模型描述

TerraMind 由 IBM Research、European Space Agency 和 Forschungszentrum Jülich 等机构提出。模型使用 TerraMesh 中约 900 万个全球时空对齐样本和约 5000 亿个训练 Token 进行预训练。模型适用于跨模态生成、地球观测表征、土地覆盖分割、水体识别、植被评估和 Thinking-in-Modalities 等任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 多模态表征 | 联合编码光学、雷达、高程及其他地球观测模态。 |
| 任意到任意生成 | 根据 Sentinel-2、坐标和文本等已有模态预测 LULC、NDVI、雷达等目标模态 Token。 |
| 双尺度学习 | 同时使用原始像素 Patch 和离散 Token 表征。 |
| 本地工程验证 | 使用对齐虚拟数据检查训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/TerraMind --local_dir ./TerraMind
cd TerraMind
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证，官方规模训练和扩散解码需要大规模加速资源。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本仓库使用时空对齐的虚拟多模态样本验证工程流程。源数据保持 TerraMesh 的 `264×264` 空间尺寸，包括 12 通道 Sentinel-2 L2A、13 通道 Sentinel-2 L1C、3 通道 RGB、2 通道 Sentinel-1 GRD、2 通道 Sentinel-1 RTC 和单通道 DEM。加载器从源样本联合裁剪官方模型使用的 `224×224` 输入，并从同一区域构造与像素内容相关的 LULC、NDVI、雷达等离散 Token，保证双尺度空间对齐。

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

默认配置只缩小样本数量、Transformer 宽度、深度、词表和训练周期，不改变源数据及模型输入空间尺寸。训练过程随机选择输入像素模态、输入 Token 模态和目标模态，并对输入 Patch 进行随机采样，以模拟论文的多模态掩码建模策略。

```text
result/checkpoints/terramind.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置虚拟权重或官方权重。IBM 和 ESA 已公开 TerraMind tiny、small、base 和 large 等版本的模型权重。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，仅以 Sentinel-2 L2A 像素、坐标 Token 和文本 Token 为条件生成 LULC、NDVI 和 Sentinel-1 GRD Token，并保存跨模态 embedding 和目标 Token：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估各目标模态的 Token 准确率和跨模态 embedding 范数，并生成目标 Token 与生成 Token 的空间对比图。虚拟数据结果仅用于验证工程链路，不代表论文中的 PANGAEA、生成质量或 Thinking-in-Modalities 指标。

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

本仓库为 TerraMind 论文公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
