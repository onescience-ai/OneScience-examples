<p align="center">
  <strong>
    <span style="font-size: 30px;">MULTI-evolve</span>
  </strong>
</p>

# 模型介绍

MULTI-evolve（model-guided, universal, targeted installation of multi-mutants）是一个面向蛋白质定向进化的端到端框架，用于训练序列-适应度预测模型、提出组合多突变体、生成 MULTI-assembly 定点诱变寡核苷酸，并支持通过蛋白质语言模型零样本集成方法筛选单点突变候选。

论文：

> **Rapid directed evolution guided by protein language models and epistatic interactions**  
> Science, 2026  
> https://doi.org/10.1126/science.aea1820

# 模型描述

MULTI-evolve 的核心流程包括：

1. 使用实验序列-适应度数据训练全连接神经网络；
2. 比较不同数据划分、序列表征和机器学习模型；
3. 选择表现最佳的预测模型，对组合突变体进行评分并提出候选；
4. 根据筛选出的多突变体生成 MULTI-assembly 定点诱变寡核苷酸；
5. 在部分迭代中使用蛋白质语言模型零样本集成方法筛选单点突变候选。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 蛋白质定向进化 | 基于实验数据训练适应度预测模型并筛选候选突变 |
| 多突变体设计 | 对组合突变进行预测并筛选高适应度多突变体 |
| 蛋白质复合物优化 | 支持多链蛋白质的突变格式和多链输入 |
| 零样本突变筛选 | 使用蛋白质语言模型集成方法筛选候选单点突变 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- MULTI-evolve 的监督模型训练和普通组合突变预测可使用 CPU 或 GPU/DCU。
- 蛋白质语言模型零样本预测涉及 ESM、ESM-IF 等模型，推荐使用 GPU/DCU。

### 安装运行环境

#### DCU环境

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311

pip install onescience[bio] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

#### 环境说明

- 在实际运行过程中，如遇到依赖缺失或版本不兼容等问题，可参考 `env.yml` 声明的依赖版本，补充安装或调整相应依赖。

### 模型与数据准备

MULTI-evolve 的不同功能对模型和数据的依赖并不相同。普通监督训练、组合多突变体推荐、MULTI-assembly 设计、蛋白质语言模型零样本筛选以及 benchmark 复现所需资源应分别准备。

#### 1）监督学习输入数据

如果使用 MULTI-evolve 训练自己的蛋白质适应度预测模型，需要准备：

```text
野生型蛋白质 FASTA
+
实验训练数据 CSV
```

训练数据 CSV 至少包含：

```text
mutation
property_value
```

例如单链蛋白质突变格式：

```text
A40P/E61Y
```

多链蛋白质使用 `:` 分隔不同链：

```text
A40P/E61Y:WT
```

其中：

```text
/   分隔同一条链上的多个突变
:   分隔不同蛋白质链
WT  表示对应链保持野生型
```

官方仓库已经提供示例数据：

```text
data/
├── example_protein/
└── example_multichain_protein/
```

因此运行官方基础示例时，不需要额外下载训练数据。

#### 2）组合突变候选池

运行组合多突变体推荐时，除野生型 FASTA 和训练数据外，还需要提供 mutation pool，即允许参与组合设计的候选单点突变列表。

示例：

```text
data/example_protein/combo_muts.csv
```

该文件作为：

```text
--mutation-pool
```

参数输入，例如：

```bash
p2_propose.py \
  --experiment-name multievolve_example \
  --protein-name example_protein \
  --wt-files apex.fasta \
  --training-dataset example_dataset.csv \
  --mutation-pool combo_muts.csv \
  --top-muts-per-load 3 \
  --export-name multievolve_proposals
```

#### 3）蛋白质语言模型零样本模式

MULTI-evolve 的蛋白质语言模型零样本集成流程需要：

```text
野生型 FASTA
+
PDB/CIF 蛋白质结构
```

当前官方代码实际使用以下模型：

```text
ESM-1v:
esm1v_t33_650M_UR90S_1
esm1v_t33_650M_UR90S_2
esm1v_t33_650M_UR90S_3
esm1v_t33_650M_UR90S_4
esm1v_t33_650M_UR90S_5

ESM-2:
esm2_t36_3B_UR50D

ESM-IF1:
esm_if1_gvp4_t16_142M_UR50
```

MULTI-evolve 通过 `fair-esm` 调用这些模型。首次运行时，如果本地不存在对应权重，`fair-esm` 会自动下载模型并缓存到 PyTorch Hub 的 checkpoint 目录。

默认缓存位置为：

```text
~/.cache/torch/hub/checkpoints/
```

其中 ESM-2 还会使用对应的 contact regression 权重：

```text
esm2_t36_3B_UR50D-contact-regression.pt
```
- 当前仓库在 `hub/checkpoints/` 下已放置 `esm2_t36_3B_UR50D-contact-regression.pt`。

对于网络受限或离线环境，建议提前下载：

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
cd ~/.cache/torch/hub/checkpoints

wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_2.pt
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_3.pt
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_4.pt
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_5.pt

wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt
wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt

wget https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt
```

如果希望将模型统一保存到当前项目或其他位置，推荐通过 `TORCH_HOME` 指定 PyTorch Hub 缓存根目录。例如当前项目内路径可设置为：

```bash
cd /path/to/MULTI-evolve
export TORCH_HOME=$PWD
mkdir -p ${TORCH_HOME}/hub/checkpoints
```

随后将上述权重保存或软链接到：

```text
/path/to/MULTI-evolve/hub/checkpoints/
```

这样无需修改 MULTI-evolve 源码。

#### 4）Benchmark DMS 数据

如果需要运行官方 benchmark，复现不同：

```text
数据划分方法
序列表征方法
机器学习模型
```

之间的性能比较，则必须额外准备官方 benchmark DMS 数据，需要从 Zenodo 单独下载。

下载地址：

```text
DOI: 10.5281/zenodo.17620759
https://zenodo.org/records/17620759
```

下载完成后，将 DMS CSV 文件直接放入以下目录；如果该目录不存在，可先手动创建：

```text
data/benchmark/datasets/
```

当前仓库的 benchmark 脚本入口位于：

```text
scripts/notebooks/benchmark/multievolve_hyperparameter_tuning.py
```

## 3. 快速开始

### 下载模型包

```bash
modelscope download \
  --model OneScience/MULTI-evolve \
  --local_dir ./MULTI-evolve

cd MULTI-evolve
```

- MULTI-evolve 普通监督训练和组合突变推荐不需要额外下载大型固定数据集，可直接使用仓库中的示例数据或自己的实验数据。

- 蛋白质语言模型零样本模式可能需要额外准备 ESM/ESM-IF 模型缓存；离线环境建议提前准备。

### 快速验证

安装当前仓库：

```bash
python -m pip install -e . --no-deps
```

检查命令：

```bash
p1_train.py --help
p2_propose.py --help
p3_assembly_design.py --help
plm_zeroshot_ensemble.py --help
```


# 示例数据

官方仓库提供：

```text
data/
├── example_protein/
├── example_multichain_protein/
└── benchmark/
```

官方命令行示例主要使用：

```bash
cd data/example_protein
```

典型输入包括：

```text
apex.fasta
example_dataset.csv
combo_muts.csv
APEX_33overhang.fasta
apex.cif
```

这些文件分别用于：

| 文件 | 用途 |
| --- | --- |
| `apex.fasta` | 野生型蛋白质氨基酸序列 |
| `example_dataset.csv` | 训练数据 |
| `combo_muts.csv` | 组合突变候选池 |
| `APEX_33overhang.fasta` | MULTI-assembly 寡核苷酸设计所需 DNA 输入 |
| `apex.cif` | 蛋白质语言模型结构条件打分 |

# 推理与训练示例

## Step 1：训练神经网络模型

```bash
# 如果尚未激活运行环境，请先激活实际使用的 conda 环境，例如 onescience311
conda activate onescience311
cd data/example_protein

p1_train.py \
  --experiment-name multievolve_example \
  --protein-name example_protein \
  --wt-files apex.fasta \
  --training-dataset-fname example_dataset.csv \
  --wandb-key dummy \
  --mode test
```

主要参数：

| 参数 | 说明 |
| --- | --- |
| `--experiment-name` | 当前实验名称，后续步骤应保持一致 |
| `--protein-name` | 蛋白质名称 |
| `--wt-files` | 野生型 FASTA；多链可用逗号分隔多个 FASTA |
| `--training-dataset-fname` | 训练数据 CSV |
| `--mode` | `test` 或 `standard` |

## Step 2：提出组合多突变体

```bash
p2_propose.py \
  --experiment-name multievolve_example \
  --protein-name example_protein \
  --wt-files apex.fasta \
  --training-dataset example_dataset.csv \
  --mutation-pool combo_muts.csv \
  --top-muts-per-load 3 \
  --export-name multievolve_proposals
```

脚本会加载 Step 1 保存到本地缓存的训练模型，并对组合突变候选进行评分。

典型输出：

```text
multievolve_proposals.csv
```

对于蛋白质复合物，还会为不同链分别生成候选文件。

## Step 3：设计 MULTI-assembly 寡核苷酸

```bash
p3_assembly_design.py \
  --mutations-file multievolve_proposals.csv \
  --wt-fasta APEX_33overhang.fasta \
  --overhang 33 \
  --species human \
  --oligo-direction top \
  --tm 80 \
  --output design
```

其中：

| 参数 | 说明 |
| --- | --- |
| `--mutations-file` | Step 2 生成的候选突变 CSV |
| `--wt-fasta` | 包含两端 overhang 的野生型 DNA FASTA |
| `--overhang` | overhang 长度 |
| `--species` | `human`、`ecoli` 或 `yeast` |
| `--oligo-direction` | `top` 或 `bottom` |
| `--tm` | 寡核苷酸目标熔解温度，官方推荐 80 °C |
| `--output` | `design` 或 `update` |

输出：

```text
cloning_sheet.csv
oligos.csv
```

## 蛋白质语言模型零样本集成

```bash
plm_zeroshot_ensemble.py \
  --wt-file apex.fasta \
  --pdb-files apex.cif \
  --chain-id A \
  --variants 24 \
  --excluded-positions 1,14,41,112 \
  --normalizing-method aa_substitution_type
```

其中：

| 参数 | 说明 |
| --- | --- |
| `--wt-file` | 野生型蛋白质 FASTA |
| `--pdb-files` | PDB/CIF 结构文件，可用逗号分隔多个结构 |
| `--chain-id` | 结构文件中目标蛋白质链的 chain ID |
| `--variants` | 每种方法提名的突变数量 |
| `--excluded-positions` | 不参与突变的位置 |
| `--normalizing-method` | `aa_substitution_type` 或 `aa_mutation` |

该流程包含 4 种方法的集成，最终输出：

```text
plm_zeroshot_ensemble_nominated_mutations.csv
```

# 输出说明

MULTI-evolve 会在不同阶段生成模型缓存、评测结果和候选序列。

官方仓库运行后会自动创建：

```text
proteins/
└── <protein_name>/
    ├── feature_cache/
    ├── model_cache/
    │   └── <dataset>/
    │       ├── objects/
    │       └── results/
    ├── proposers/
    │   └── results/
    └── split_cache/
        └── <dataset>/
```

主要输出包括：

| 输出 | 说明 |
| --- | --- |
| `model_cache/` | 训练得到的模型及比较结果 |
| `feature_cache/` | 缓存后的序列表征 |
| `multievolve_proposals.csv` | 推荐的多突变候选 |
| `cloning_sheet.csv` | MULTI-assembly 克隆设计表 |
| `oligos.csv` | 定点诱变寡核苷酸序列 |
| `plm_zeroshot_ensemble_nominated_mutations.csv` | 蛋白质语言模型零样本集成推荐结果 |


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |


# 引用与许可证

- MULTI-evolve 原始论文：[Rapid directed evolution guided by protein language models and epistatic interactions](https://doi.org/10.1126/science.aea1820)。
- 仓库根目录 `LICENSE` 当前为 **Apache License 2.0**。该许可证允许使用、修改、分发和商业使用，但再分发时需要保留许可证、版权与归属声明，并对修改过的文件进行明显说明。
- Apache-2.0 同时包含专利许可条款，并明确不授予项目商标使用权。
- `setup.py` 中当前仍存在 `MIT License` classifier，与仓库根目录实际 `LICENSE` 文件不一致；进行 SCNet/ModelScope 再分发时应以仓库根目录的 Apache-2.0 `LICENSE` 为准，并建议保留原始许可证文件。
- 本仓库为 MULTI-evolve 的 DCU 适配版本，对部分环境配置、依赖及运行方式进行了调整；仓库代码、模型权重及相关数据的使用仍应以各自原始项目中的许可证及使用条款为准。
