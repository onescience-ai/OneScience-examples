<p align="center">
  <strong><span style="font-size: 30px;">CausalModelEvaluation</span></strong>
</p>

# 模型介绍

CausalModelEvaluation（CME）用于解决传统气候模型评价过度依赖平均状态误差、难以衡量过程关系是否正确的问题，通过比较气候模型与再分析资料中的因果联系评价模型对关键气候过程的表示能力。该方法主要用于构建气候因果指纹、识别具有共同开发背景的模型依赖性、评价降水模拟能力，并利用历史过程技能约束未来降水变化的不确定性。

论文：Causal networks for climate model evaluation and constrained projections  
https://doi.org/10.1038/s41467-020-15195-y

# 模型描述

Causal Model Evaluation 由 Imperial College London、German Aerospace Center、University of Bremen 和 University of East Anglia 的研究团队提出。论文使用 CMIP5 海平面气压与降水模拟、NCEP-NCAR 和 ERA-Interim 再分析以及 CRU TS v4.02 降水观测开展评价。模型适用于气候因果指纹重建、过程导向的气候模型评价和降水变化约束投影。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 因果指纹重建 | 从 50 节点季节时间序列估计有向、有符号、时滞网络。 |
| 气候模型评估 | 以参考网络为基准计算方向、符号和时滞容忍的非对称 F1。 |
| 降水技能评估 | 计算带纬度面积权重的 Taylor S-score 和空间型相关。 |
| 约束投影 | 拟合 CME F1 与降水变化之间的 RBF 加白噪声 GP。 |
| ModelScope/OneCode | 验证数据、拟合、checkpoint、推理、评估和任务图流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/CausalModelEvaluation --local_dir ./CausalModelEvaluation
cd CausalModelEvaluation
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

```bash
python scripts/fake_data.py
```

本仓库使用少量结构化虚拟样本验证工程流程，数据包含四季海平面气压模态时间序列、模型间不同的滞后因果关系，以及与模型过程技能相关的降水空间场和降水变化。虚拟数据保持论文的节点、时间、时滞和空间维度，仅用于验证因果网络构建、模型比较和约束投影流程，不代表真实气候数据分布与论文性能。

### 训练

单卡训练可使用：

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练结果包含参考与模型因果网络、模型比较分数和降水约束关系，并保存 checkpoint 与训练指标供后续推理使用。

```text
result/checkpoints/causalmodelevaluation.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置权重。论文方法不产生传统神经网络权重，工程 checkpoint 保存统计网络与 GP 状态，不声明兼容外部权重。

### 推理

```bash
python scripts/inference.py
```

推理重新加载 checkpoint，输出所有模型与四季的完整 `edges/pvalues/mci`、完整参考网络、参考及模型降水场、CME F1、降水变化、GP 均值与 95% 区间，以及网络和投影元数据到 `result/output/inference.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估生成气候模型因果网络比较、降水模拟能力和约束投影的结构化结果，并保存到 `result/evaluation/metrics.json`。脚本同时生成模型技能与降水变化关系的对比图 `result/evaluation/cme_task.png`。虚拟数据结果仅用于验证工程流程，不代表论文真实测试集性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 Causal Model Evaluation 论文公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
