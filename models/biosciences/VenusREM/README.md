<p align="center">
  <strong>
    <span style="font-size: 30px;">VenusREM</span>
  </strong>
</p>

# 模型介绍

VenusREM 是一个面向蛋白质突变效应预测的零样本模型。它以 ProSST 的序列—结构语言模型 logits 为基础，引入由同源序列比对得到的进化信息，通过 retrieval-based logits fusion 计算候选突变的适应度分数。

VenusREM 可用于单点或由多个替换组成的突变评分。输出分数用于比较候选突变相对于野生型的模型偏好，不等同于具有统一物理单位的实验测量值。

论文：[From high-throughput evaluation to wet-lab studies: advancing mutation effect prediction with a retrieval-enhanced model](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i401/8199372)

# 模型描述

VenusREM 的基础模型采用 ProSST-2048：

- 使用氨基酸序列表示蛋白质一级结构；
- 将每个残基的局部三维环境量化为 2048 类结构 token；
- 通过序列—结构联合建模生成每个位置的氨基酸 logits；
- 从 A2M/A3M/FASTA 同源序列比对中统计进化分布；
- 使用参数 `alpha` 融合语言模型 logits 与检索到的进化 logits；
- 根据突变氨基酸与野生型氨基酸的 log-probability 差异输出突变分数。



# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 零样本突变排序 | 无需目标蛋白监督标签，对候选氨基酸替换进行相对排序 |
| ProteinGym 评测 | 在 ProteinGym substitutions 数据上生成逐蛋白分数和 Spearman 相关性汇总 |
| MSA 增强预测 | 融合同源序列比对中的进化信息，提高突变评分的上下文相关性 |
| ProSST 基线预测 | 设置 `alpha=0`，仅使用序列—结构语言模型 logits |
| 结构 token 生成 | 在只有 PDB、没有预计算结构 token 时执行可选预处理 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 支持 CPU 和 OneScience DTK 环境中的 DCU；
- 推荐使用 DCU 执行完整 ProteinGym 推理和 PDB 到结构 token 的预处理；
- CPU 可以完成基础验证，但完整数据集和结构预处理耗时明显更长；


### 下载模型包

安装 ModelScope 命令行工具后下载模型：

```bash
python -m pip install modelscope
modelscope download --model OneScience/VenusREM --local_dir ./VenusREM
cd VenusREM
```

### 安装运行环境

**DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
python -m pip install "onescience[bio-dcu]" \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

在 OneScience 环境基础上安装 VenusREM 的额外依赖：

```bash
python -m pip install --no-deps -r requirements.txt
```



### 权重与数据准备

模型推理所需的官方 ProSST-2048 资产位于：

| 资产 | 相对位置 | 用途 |
| --- | --- | --- |
| 模型权重 | `weight/ProSST-2048/model.safetensors` | ProSST-2048 参数 |
| 模型配置 | `weight/ProSST-2048/config.json` | 网络结构与词表配置 |
| 模型实现 | `weight/ProSST-2048/modeling_prosst.py` | Transformers 自定义模型实现 |
| 配置实现 | `weight/ProSST-2048/configuration_prosst.py` | Transformers 自定义配置实现 |
| 氨基酸词表 | `weight/ProSST-2048/vocab.txt` | 序列分词 |
| Tokenizer 配置 | `weight/ProSST-2048/tokenizer_config.json` | Tokenizer 参数 |

推理数据集使用以下相对目录结构：

```text
conf/data/<dataset_name>/
├── aa_seq/
│   └── protein1.fasta
├── aa_seq_aln_a2m/
│   └── protein1.a2m
├── struc_seq/
│   └── 2048/
│       └── protein1.fasta
└── substitutions/
    └── protein1.csv
```

文件主名必须一致，例如 `protein1.fasta`、`protein1.a2m` 和 `protein1.csv` 对应同一个蛋白质。`substitutions/*.csv` 至少需要包含：

- `mutant`：突变表示，如 `A10V`；多个替换使用冒号分隔，如 `A10V:G25D`；
- `DMS_score`：用于评测 Spearman 相关性。没有实验标签时可填 0，但此时相关性结果没有评测意义。

结构 token 是基础模型输入的一部分，即使设置 `alpha=0` 不使用 MSA，也需要准备 `struc_seq/2048/*.fasta`。

### 快速推理

**作用：** 该模式融合 ProSST-2048 的序列—结构 logits 与残基序列 MSA 中的进化信息，适用于 ProteinGym 全量评测，以及具备预计算 A2M/A3M 比对数据的正式突变候选排序。

以下命令使用 ProSST-2048 权重和预计算的残基序列 MSA，在 DCU 上执行 VenusREM 推理：

```bash

python scripts/compute_fitness.py \
  --model_name weight/ProSST-2048 \
  --model_out_name VenusREM_DCU \
  --base_dir conf/data/proteingym_v1 \
  --out_scores_dir output/proteingym_v1 \
  --logit_mode aa_seq_aln \
  --alpha 0.8
```

输出文件为：

```text
output/proteingym_v1/
├── scores/
│   └── <protein_name>.csv
└── summary_performance.csv
```

每个蛋白质的 CSV 会新增 `VenusREM_DCU` 分数列。`summary_performance.csv` 记录各蛋白质数据集的 Spearman 相关性。



### 不使用 MSA 的 ProSST 推理

**作用：** 运行不包含检索增强的 ProSST-2048 基线，用于比较 MSA 融合前后的突变评分差异、开展消融分析，或在没有同源序列比对文件时完成基础序列—结构突变评分。

设置 `alpha=0` 后，不融合残基或结构比对 logits，但仍需氨基酸序列、结构 token 和 substitutions 文件：

```bash
python scripts/compute_fitness.py \
  --model_name weight/ProSST-2048 \
  --model_out_name ProSST-2048 \
  --base_dir conf/data/proteingym_v1 \
  --out_scores_dir output/prosst_2048 \
  --alpha 0
```

### 使用结构序列比对

**作用：** 使用 Foldseek 结构序列比对产生的结构同源信息增强 ProSST logits。该模式适用于已经准备结构比对结果、希望评估结构检索信息贡献，或需要与残基序列 MSA 模式进行对照的场景。

如果已经准备 Foldseek 结构序列比对，可执行：

```bash
python scripts/compute_fitness.py \
  --model_name weight/ProSST-2048 \
  --model_out_name VenusREM_struc \
  --base_dir conf/data/proteingym_v1 \
  --out_scores_dir output/proteingym_v1_struc \
  --logit_mode struc_seq_aln \
  --alpha 0.8
```

对应的比对文件应位于：

```text
conf/data/proteingym_v1/struc_seq_aln_foldseek/<protein_name>.fasta
```

### PDB 转结构 token

**预处理作用：** 将蛋白质 PDB 中每个残基的局部三维环境量化为 ProSST-2048 可读取的结构 token。生成结果是上述所有推理模式的结构输入，不直接输出突变适应度分数。

只有 PDB、没有 `struc_seq/2048/*.fasta` 时，才需要执行该预处理。已有官方预计算结构 token 时可以跳过。

单个 PDB：

```bash
python model/data/get_struc_seq.py \
  --pdb_file conf/data/proteingym_v1/pdbs/protein1.pdb \
  --output_dir conf/data/proteingym_v1/struc_seq \
  --vocab_size 2048 \
  --overwrite
```

批量 PDB：

```bash
python model/data/get_struc_seq.py \
  --pdb_dir conf/data/proteingym_v1/pdbs \
  --output_dir conf/data/proteingym_v1/struc_seq \
  --vocab_size 2048 \
  --overwrite
```



### 自定义数据推理

**作用：** 对用户自己的蛋白质和候选突变进行零样本适应度评分，用于湿实验前的候选排序、蛋白质工程初筛或自定义数据集评测。输入文件名必须在氨基酸序列、MSA、结构 token 和 substitutions 目录之间保持一致。

建立与前述格式一致的相对目录，例如：

```text
conf/data/my_proteins/
├── aa_seq/
├── aa_seq_aln_a2m/
├── struc_seq/2048/
└── substitutions/
```

然后运行：

```bash
python scripts/compute_fitness.py \
  --model_name weight/ProSST-2048 \
  --model_out_name VenusREM \
  --base_dir conf/data/my_proteins \
  --out_scores_dir output/my_proteins \
  --logit_mode aa_seq_aln \
  --alpha 0.8
```

如果没有 substitutions 文件，可使用官方辅助脚本生成所有单点替换候选，再根据实际需求筛选：

```bash
python model/data/get_sav.py \
  --fasta_file conf/data/my_proteins/aa_seq/protein1.fasta \
  --output_csv conf/data/my_proteins/substitutions/protein1.csv
```

### 训练

VenusREM 官方仓库没有提供可直接运行的训练入口、Dataset、优化器或完整训练循环，所以本模型包不提供训练脚本。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- VenusREM 论文：[From high-throughput evaluation to wet-lab studies: advancing mutation effect prediction with a retrieval-enhanced model](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i401/8199372)
- 官方实现：[ai4protein/VenusREM](https://github.com/ai4protein/VenusREM)
- 基础模型：[ai4protein/ProSST](https://github.com/ai4protein/ProSST)
- 本项目依据 `CC-BY-NC-ND-4.0` 许可证提供，使用模型、代码、数据和第三方资产时应同时遵守各自的许可证与使用条款。
