<p align="center">
  <strong><span style="font-size: 30px;">IMPROVER-AIFS</span></strong>
</p>

# 模型介绍

IMPROVER-AIFS 用于订正人工智能天气模型的系统偏差并改善预报可靠性，将确定性 AIFS 预报转换为经过空间处理和统计校准的确定性与概率产品，并可与传统数值天气预报结果融合，主要用于近地面天气要素后处理和业务预报产品生成。

论文：Statistical Postprocessing Yields Accurate Probabilistic Forecasts from Artificial Intelligence Weather Models  
https://doi.org/10.1175/AIES-D-25-0037.1

# 模型描述

IMPROVER-AIFS 对应的方法由 Australian Bureau of Meteorology 的研究团队提出，并采用由 Met Office 主导开发的 IMPROVER 后处理系统。论文使用 ECMWF AIFS、HRES 和 ENS 预报、MSAS 网格分析以及 Bureau Jive 自动气象站观测开展校准和评估。模型适用于地表温度、地表露点温度和 10 米风速的确定性订正、概率校准与多模型融合预报。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 确定性预报后处理 | 对 AIFS 类天气预报执行高度与历史偏差订正。 |
| 概率预报与校准 | 生成三变量阈值概率，并执行空间平滑和可靠性校准。 |
| 多模型融合 | 平滑融合 AIFS、HRES 和 ENS 类输入的确定性与概率结果。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/IMPROVER-AIFS --local_dir ./IMPROVER-AIFS
cd IMPROVER-AIFS
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
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

本仓库使用少量结构化虚拟样本验证工程流程，数据包含 AIFS、HRES 和 ENS 三类预报，以及对应的 MSAS 分析和 Bureau Jive 站点观测语义。每个样本保留 30 天历史、241 个逐小时时效、三个变量、`61/47/49` 个阈值、全部 569 个站点及每站 `3×3` 邻域，只缩小有效日期数量和参与参数拟合的站点样本数。论文未公开完整目标 Albers 网格尺寸，因此本实现不推断该维度；虚拟数据仅用于验证后处理、训练、推理和评估流程，不代表论文数据分布与性能。

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
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练产物保存到：

```text
result/checkpoints/improver_aifs.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。论文未提供可确认的官方 checkpoint，当前工程 checkpoint 不声明兼容外部权重。

### 推理

```bash
python scripts/inference.py
```

推理加载包含两个 valid-time fold 的 checkpoint，按站点分块生成两个有效日期、241 个时效和全部 569 站的确定性及概率融合结果。数值结果及站点元数据保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估结果包含原始预报、后处理预报和融合预报的误差与概率预报质量，并生成代表性时效的目标、预测和误差对比图。虚拟数据结果仅用于验证工程流程，不代表论文真实性能。

```text
result/evaluation/metrics.json
result/evaluation/multi_lead_temperature.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 IMPROVER-AIFS 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
