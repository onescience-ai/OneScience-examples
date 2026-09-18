<p align="center">
  <strong><span style="font-size: 30px;">ClimEmu-S2L</span></strong>
</p>

# 模型介绍

ClimEmu-S2L 根据气候强迫情景前 10 年的全球近地表温度响应预测长期气候变化空间格局，以降低多情景长期气候模拟的计算成本并支持区域温度响应分析。

论文：Predicting global patterns of long-term climate change from short-term simulations using machine learning  
https://doi.org/10.1038/s41612-020-00148-5

# 模型描述

ClimEmu-S2L 对应的方法由 Imperial College London、University of Reading、University of East Anglia、University of Warwick 和 Technical University of Crete 等机构的研究团队提出。论文使用 HadGEM3 在 PDRMIP、ECLIPSE 和 Kasoar 等计划中的 21 个气候强迫情景模拟数据。模型执行由短期全球近地表温度响应预测长期空间格局的任务，并评估全球和区域气候响应。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 长期气候响应预测 | 根据前 10 年温度响应预测 70 年后的长期全球空间格局。 |
| Ridge 回归验证 | 使用内层交叉验证选择正则化参数并执行 21 折 LOSO 预测。 |
| GPR 回归验证 | 使用共享核高斯过程回归执行 21 折 LOSO 预测。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中执行完整网格训练、推理、分区域评估和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 将 21 个 LOSO fold 分配到多个进程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/ClimEmu-S2L --local_dir ./ClimEmu-S2L
cd ClimEmu-S2L
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

本仓库使用少量结构化虚拟气候响应样本验证工程流程，数据包含 21 个强迫情景在完整 `145×192` 网格上的单通道近地表温度异常，单位为 `degC`。输入是每个情景前 10 年的平均响应，目标是同一情景 70 年后的长期平均响应。虚拟数据保留情景数量、空间网格和时间窗口，仅将 GPR 缩小为共享非 ARD 核结构并减少核优化迭代，以控制工程验证成本；这些数据不代表 HadGEM3 官方数据分布和训练规模。

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

训练分别拟合 Ridge 和 GPR，并完成 21 个留一情景交叉验证（LOSO）fold；Ridge 使用内层 3 折交叉验证选择正则化参数，GPR 优化共享非 ARD 核。标准训练产物保存到：

```text
result/checkpoints/climemu_s2l.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。论文未提供可确认的官方 checkpoint，当前工程 checkpoint 不声明为论文正式模型参数。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，并为 21 个情景生成 Ridge 和 GPR 长期响应场。完整数值结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估按方法、每个情景和每个区域报告面积加权全网格 RMSE、全球平均绝对误差及区域平均绝对误差；该任务不是多步预测或分类任务，因此不提供 per step 或 per class 结果。可视化包含目标与预测的空间图以及跨情景误差箱线图，结构化结果和图片保存到以下路径。虚拟数据结果仅用于验证工程流程，不代表论文正式性能。

```text
result/evaluation/metrics.json
result/evaluation/spatial_fields.png
result/evaluation/error_boxplots.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 ClimEmu-S2L 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
