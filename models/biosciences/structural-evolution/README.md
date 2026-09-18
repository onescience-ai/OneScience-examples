<p align="center">
  <strong>
    <span style="font-size: 30px;">Structural Evolution</span>
  </strong>
</p>

# 模型介绍

Structural Evolution 是一个基于结构信息蛋白质语言模型的无监督蛋白质与抗体突变推荐流程。该方法以蛋白质或蛋白质复合物的 PDB/CIF 结构为输入，利用 ESM-IF1 对结构条件下的候选突变序列进行打分，并从深度突变扫描候选中筛选出高概率替换，用于辅助蛋白质和抗体序列优化。

论文：

> **Unsupervised evolution of protein and antibody complexes with a structure-informed language model**  
> https://doi.org/10.1126/science.adk8946

# 模型描述

Structural Evolution 使用结构条件语言模型 ESM-IF1 对给定蛋白质结构中的目标链进行突变推荐。程序首先从输入 PDB/CIF 中提取目标链原始序列，生成单点深度突变扫描候选序列，然后在单链或多链骨架条件下计算候选序列的 log-likelihood，并按照得分从高到低筛选推荐突变。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 蛋白质单点突变推荐 | 从结构出发筛选高概率氨基酸替换 |
| 抗体序列优化 | 分别对抗体重链、轻链进行结构条件突变推荐 |
| 蛋白质复合物优化 | 在多链骨架上下文中评估目标链突变 |
| 深度突变扫描候选筛选 | 对全部单点突变进行打分并输出高分候选 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- Structural Evolution 支持 CPU 和 GPU 运行。
- ESM-IF1 推理推荐使用 GPU/DCU，加速候选序列打分。
- 对较大蛋白质复合物或大规模候选突变库，GPU/DCU 显存与主机内存占用会增加。
- 如果当前环境没有可用 GPU/DCU，也可通过 `--nogpu` 显式使用 CPU，但运行速度会明显下降。

### 安装运行环境

#### DCU环境

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311

# 支持uv安装
pip install onescience[bio] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

#### 环境说明

- 在实际运行过程中，如遇到依赖缺失或版本不兼容等问题，可参考 `environment.yml` 声明的依赖版本，补充安装或调整相应依赖。

### 权重准备

Structural Evolution 完整推理只需要额外准备 **ESM-IF1 模型权重**。普通突变推荐不需要额外下载论文训练数据集。

#### 1）ESM-IF1 模型权重

下载 ESM-IF1 权重：

```text
https://zenodo.org/records/12631662
```

命令：

```bash
wget -P ~/.cache/torch/hub/checkpoints \
  https://zenodo.org/records/12631662/files/esm_if1_20220410.zip

unzip ~/.cache/torch/hub/checkpoints/esm_if1_20220410.zip \
  -d ~/.cache/torch/hub/checkpoints/
```

解压后需要保证：

```text
~/.cache/torch/hub/checkpoints/
└── esm_if1_20220410.pt
```

存在。

`scripts/recommend.py` 会固定从：

```text
~/.cache/torch/hub/checkpoints/esm_if1_20220410.pt
```

加载模型，因此如果将权重保存到其他位置，需要修改代码中的 checkpoint 路径。

## 3. 快速开始

### 下载模型包

```bash
modelscope download \
  --model OneScience/structural-evolution \
  --local_dir ./structural-evolution

cd structural-evolution
```

- Structural Evolution 额外依赖 **ESM-IF1 模型权重**；请先按照“权重准备”完成 `esm_if1_20220410.pt` 的准备。保证代码中的 checkpoint 路径与实际保存位置一致。

### 快速验证

查看推理参数：

```bash
python scripts/recommend.py --help
```

使用官方示例结构进行快速验证：

```bash
python scripts/recommend.py \
  scripts/examples/7mmo_abc_fvar.pdb \
  --chain A \
  --n 10
```

# 示例数据

官方仓库在：

```text
scripts/examples/
```

目录提供示例结构：

```text
scripts/examples/
└── 7mmo_abc_fvar.pdb
```

该结构包含 LYCoV-1404 抗体可变区及 SARS-CoV-2 RBD，用于演示如何对抗体重链进行突变推荐。

对于自己的任务，需要准备：

```text
PDB 或 CIF 蛋白质/蛋白质复合物结构
+
目标链 ID
```

例如：

```text
structure.pdb
chain A
```

# 推理示例

## 基础突变推荐

最简单的运行方式：

```bash
python scripts/recommend.py \
  /path/to/structure.pdb \
  --chain A
```

默认输出：

```text
Top 10 mutations
maxrep = 1
multichain backbone = True
```

也就是默认推荐 10 个突变，并保证同一原始残基位点最多出现一次。

## 抗体示例

官方示例：

```bash
python scripts/recommend.py \
  scripts/examples/7mmo_abc_fvar.pdb \
  --chain A \
  --seqpath scripts/examples/7mmo_chainA_lib.fasta \
  --outpath scripts/examples/7mmo_chainA_scores.csv \
  --upperbound 109 \
  --offset 1
```

其中：

| 参数 | 说明 |
| --- | --- |
| `--chain` | 目标链 ID |
| `--seqpath` | 生成的深度突变扫描 FASTA 输出路径 |
| `--outpath` | 所有突变候选得分 CSV 输出路径 |
| `--n` | 最终推荐突变数量，默认 10 |
| `--maxrep` | 同一位点在推荐结果中最大出现次数，默认 1 |
| `--upperbound` | 最终推荐时只考虑小于该残基编号的位置 |
| `--offset` | PDB 残基编号偏移修正 |
| `--order` | 多链结构中指定链顺序 |
| `--multichain-backbone` | 使用全部链骨架作为结构条件 |
| `--singlechain-backbone` | 仅使用目标链骨架 |
| `--nogpu` | 强制使用 CPU |

上述抗体示例中，`--chain A` 指定重链；`--upperbound 109` 用于排除最终 framework region 中的突变；由于输入结构缺少第一个残基，使用 `--offset 1` 修正突变编号。

## 自定义推荐数量

例如输出前 20 个候选，并允许同一位点最多出现 2 次：

```bash
python scripts/recommend.py \
  /path/to/your_structure.pdb \
  --chain A \
  --n 20 \
  --maxrep 2
```

## 单链骨架条件

如果只希望使用目标链自身的骨架信息：

```bash
python scripts/recommend.py \
  /path/to/your_structure.pdb \
  --chain A \
  --singlechain-backbone
```

## CPU 推理

```bash
python scripts/recommend.py \
  /path/to/your_structure.pdb \
  --chain A \
  --nogpu
```

# 输出说明

Structural Evolution 会首先根据输入结构中的目标链生成完整单点深度突变扫描序列库，然后使用 ESM-IF1 计算每个候选序列的结构条件 log-likelihood。

如果没有手动指定：

```text
--seqpath
--outpath
```

程序会自动将结果保存到 `output/` 下。

主要输出包括：

```text
*.fasta
*.csv
```

其中：

| 输出 | 说明 |
| --- | --- |
| DMS FASTA | 包含野生型序列及所有单点突变候选序列 |
| scores CSV | 保存候选序列及对应 `log_likelihood` 分数 |
| 终端输出 | 根据 `log_likelihood` 排序后的 Top-N 推荐突变 |

CSV 中的候选序列按照：

```text
log_likelihood
```

从高到低排序，最终结合 `n`、`maxrep`、`upperbound` 等参数筛选推荐突变。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Structural Evolution 原始论文：[Unsupervised evolution of protein and antibody complexes with a structure-informed language model](https://doi.org/10.1126/science.adk8946)。
- Structural Evolution 采用 **MIT License**，允许在保留版权声明和许可证文本的前提下进行使用、复制、修改、发布、分发、再许可及商业使用，详见仓库根目录 `LICENSE`。
- Structural Evolution 推理依赖 ESM-IF1 模型及 ESM 相关代码。Structural Evolution 仓库的 MIT License 不自动覆盖第三方模型权重和依赖资源；如需进行商业使用或再分发等用途，应同时核对 ESM/ESM-IF1 对应的许可证及模型权重使用条款。
- 如果在科研工作中使用本仓库，建议引用 Structural Evolution 原始论文，并根据实际使用情况补充引用 ESM-IF1 等相关工作。
