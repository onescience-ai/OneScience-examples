<p align="center">
  <strong><span style="font-size: 30px;">AtmosphericDA-DModel</span></strong>
</p>

# 模型介绍

AtmosphericDA-DModel 用于学习和修正大气数值动力模型的系统误差，将知识模型与论文的 D model 误差模型结合，以改善数据同化分析和短中期预报。

论文：Using machine learning to correct model error in data assimilation and forecast applications  
https://doi.org/10.1002/qj.4116

# 模型描述

AtmosphericDA-DModel 对应的方法由 CEREA、École des Ponts ParisTech、EDF R&D 和 European Centre for Medium-Range Weather Forecasts 的研究团队提出。论文使用带参数扰动误差的双层二维准地转通道模型，以及由完整无噪和稀疏含噪观测产生的数据同化分析轨迹训练误差修正模型。模型适用于动力模型误差学习、混合代理预报和数据同化分析改进。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 动力模型误差研究 | 研究双层流函数通道动力的参数扰动误差。 |
| 混合预报验证 | 验证扰动动力模型与 Dense D 模型分析增量的组合。 |
| 数据同化实验 | 使用完整稀疏观测窗口生成可执行近似分析；动力和观测接口可替换为完整 4D-Var。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证训练、推理、评估、可视化和 checkpoint 流程。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/AtmosphericDA-DModel --local_dir ./AtmosphericDA-DModel
cd AtmosphericDA-DModel
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

本仓库使用少量结构化双层准地转流函数样本验证工程流程，每个样本保留完整 `2×20×40` 状态，以及 12 批、每批 50 个随机位置的双线性观测。输入为分析状态，目标为下一分析状态与原始模型预报之间的误差增量；默认仅缩小样本数量，不缩减状态和观测窗口维度。虚拟数据仅用于验证模型误差学习、训练、推理和评估流程，不代表论文数据分布与正式性能。

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

训练校验状态、观测和目标格式，并使用 Adam 与 MSE 拟合模型。默认配置缩短训练轮数以快速验证流程，结果保存到：

```text
result/checkpoints/d_model.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。论文未提供可确认的官方 checkpoint，当前工程 checkpoint 不声明兼容外部权重。

### 推理

```bash
python scripts/inference.py
```

推理结果包含真实分析状态、原始模型预报、误差修正结果和混合模型预报。完整数值结果保存到 `result/output/predictions.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估报告动力模型、混合模型相对下一分析的流函数 RMSE，以及严格分析增量 RMSE，并验证完整维度和有限数值。结构化结果和图片保存到 `result/evaluation/metrics.json` 与 `result/evaluation/comparison.png`。虚拟数据结果仅验证工程流程，不代表论文正式性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 AtmosphericDA-DModel 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
