<p align="center">
  <strong><span style="font-size: 30px;">KGML-AgCarbon</span></strong>
</p>

# 模型介绍

KGML-AgCarbon 是面向农业生态系统碳循环量化的知识引导机器学习模型，通过融合过程机理、观测数据与 GRU 网络预测碳通量和作物生产指标。

论文：Knowledge-guided machine learning can improve carbon cycle quantification in agroecosystems  
https://doi.org/10.1038/s41467-023-43860-5

# 模型描述

该方法由明尼苏达大学、伊利诺伊大学厄巴纳-香槟分校和劳伦斯伯克利国家实验室等机构的研究团队提出。论文使用 ecosys 过程模型生成的合成数据、遥感 GPP、USDA 县级产量数据和农田涡度协方差通量塔观测开展预训练与微调。模型执行日尺度 GPP、Reco 和 NEE 碳通量以及年度作物产量、残体和土壤碳变化相关预测任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 农业碳通量建模 | 根据日尺度环境与作物特征预测 GPP、Reco 和 NEE。 |
| 作物产量与残体预测 | 联合预测作物产量和残体，并施加碳质量平衡与范围约束。 |
| 核心方法验证 | 验证知识引导层次结构、Basis 预训练、联合训练和生物地球化学约束损失。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证数据生成、训练、推理、评估和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/KGML-AgCarbon --local_dir ./KGML-AgCarbon
cd KGML-AgCarbon
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活 DTK 及 Conda
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活 Conda
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

模型输入为覆盖完整年度的日尺度张量 `[B,365,19]`。监督目标包含日尺度 `[GPP, Reco, NEE]` 输出以及每个样本的 yield 和 residue 输出。本仓库提供的虚拟少样本仅保留季节结构和数据契约，不代表论文真实数据分布、训练规模或正式性能。

```bash
python scripts/fake_data.py
```

### 训练

单卡训练可使用：

```bash
python scripts/train.py
```

多卡训练可使用标准 8 卡命令：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练依次执行 Basis 预训练和联合训练，并结合 MSE、`NEE = Ra + Rh - GPP` 质量平衡、yield 范围与残体约束。默认配置用于小样本工程验证，正式实验需要论文所述真实数据和完整计算资源，训练产物保存到：

```text
result/checkpoints/kgml_agcarbon.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置官方权重，也未发现可确认的官方预训练 checkpoint。论文提供的完整 Zenodo 资源 https://doi.org/10.5281/zenodo.10155516 包含代码与 Source Data，但不能据此将其中内容认定为已确认的预训练 checkpoint；本地 checkpoint 仅是本仓库虚拟数据训练生成的工程产物。

### 推理

```bash
python scripts/inference.py
```

推理加载本地训练 checkpoint，并将日尺度碳通量、yield 和 residue 预测保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估报告 R2、RMSE、质量平衡和 yield 指标，并生成对比图：

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

本仓库为 KGML-AgCarbon 公开规格的独立工程复现版本。本仓库代码的使用应遵循 Apache-2.0 许可证条款。

原论文、官方 Zenodo 代码与 Source Data、其他第三方数据或权重的引用和使用应遵循各自项目公布的许可证及使用条款；本仓库不重新授权第三方内容。
