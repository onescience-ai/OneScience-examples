<p align="center">
  <strong><span style="font-size: 30px;">FireCubeNet</span></strong>
</p>

# 模型介绍

FireCubeNet 用于解决大面积野火的次日危险预测问题，综合气象、植被、土壤湿度、人类活动、地形和土地覆盖条件判断可能发生严重野火的区域。模型主要用于生成野火危险概率、识别影响火灾发生与传播的重要因素，并为野火预警、风险评估和应急资源部署提供数据驱动支持。

论文：Wildfire Danger Prediction and Understanding With Deep Learning  
https://doi.org/10.1029/2022GL099368

# 模型描述

FireCubeNet 由 National Observatory of Athens、Universitat de València、Max Planck Institute for Biogeochemistry 和 Universidade Nova de Lisboa 的研究团队提出。论文使用 ERA5-Land、MODIS、European Drought Observatory、WorldPop、Copernicus EU-DEM、CORINE Land Cover、EFFIS 和 MODIS active fire 数据训练与验证。模型适用于次日野火危险预测、时空火灾驱动建模和大面积野火风险评估。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 次日野火危险 | 从中心像素过去 10 天及其 25 km 邻域估计次日大火危险概率。 |
| 时空驱动建模 | 联合使用天气、植被、湿度、社会经济、地形与土地覆盖变量。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 运行分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/FireCubeNet --local_dir ./FireCubeNet
cd FireCubeNet
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

默认虚拟数据仅含少量样本，但保持 `10×25×25×25` 的真实维度。生成器包含持续天气、逐日干燥、空间热点及变量间物理相关；静态空间变量在时间上重复，10 个土地覆盖 fraction 在每个像素严格归一化为 1。虚拟标签只用于工程连通性验证，不代表官方数据分布和论文性能。

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

训练结果包含用于推理的模型参数，以及各训练轮次的损失指标。训练结果保存到：

```text
result/checkpoints/firecubenet.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 内置权重。论文未提供可确认的官方 checkpoint，当前工程 checkpoint 不声明兼容外部权重。

### 推理

```bash
python scripts/inference.py
```

推理结果包含次日野火危险概率、真实标签以及对应的时间和空间位置信息，并保存到 `result/output/predictions.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估结果包含 Precision、Recall、F1、AUROC 和混淆矩阵，并保存到 `result/evaluation/metrics.json`。脚本同时生成野火危险概率与 ROC 对比图 `result/evaluation/wildfire_danger.png`。虚拟数据结果仅用于验证工程流程，不代表论文真实测试集性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 FireCubeNet 论文公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
