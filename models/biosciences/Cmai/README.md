<p align="center">
  <strong>
    <span style="font-size: 30px;">Cmai</span>
  </strong>
</p>

# 模型介绍

Cmai 是一个面向 B 细胞受体（BCR）与抗原结合亲和力预测的深度学习推理流程。该项目支持从输入 BCR/抗原序列开始，完成输入检查、抗原结构嵌入生成、BCR 表征、结合分数预测以及背景分布 rank% 计算。

论文：

> **Profiling antigen-binding affinity of B cell repertoires in tumors by deep learning predicts immune-checkpoint inhibitor treatment outcomes**  
> https://doi.org/10.1038/s43018-025-01001-5

# 模型描述

Cmai 将抗原嵌入、BCR V 区与 CDR3 区表征以及双向背景排名结合起来，用于评估候选 BCR-抗原配对的结合倾向。抗原侧依赖 RoseTTAFold 生成结构相关嵌入，BCR 侧使用仓库内置的 V/CDR3 编码模型，最终通过训练好的 PyTorch 权重输出结合分数与 rank%。


# 适用场景

| 场景 | 说明 |
| --- | --- |
| BCR-抗原结合预测 | 输入 BCR 与抗原序列，预测结合分数和背景排名 |
| 抗体设计与优化 | 以目标抗原为 anchor，在候选 BCR 中筛选更可能结合的序列 |
| 抗原发现与疫苗研究 | 以目标 BCR 为 anchor，在候选抗原中评估结合倾向 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 抗原 embedding 阶段建议使用 GPU；RoseTTAFold 在完整推理中通常需要较大显存。
- MSA 生成阶段主要使用 CPU。
- 结合预测阶段依赖 PyTorch，推荐使用 GPU 运行。
- 如果 GPU 资源有限，可先用 CPU 单独运行 MSA，再运行 RoseTTAFold 与结合预测阶段。

### 安装运行环境

#### DCU环境
```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```
#### 环境说明

- 在实际运行过程中，如遇到依赖缺失或版本不兼容等问题，可参考 `runBind.yml`、`runEmbed.yml` 以及 RoseTTAFold 官方环境配置中声明的依赖版本，补充安装或调整相应依赖。
- 用户也可分别基于 `runBind.yml` 和 `runEmbed.yml` 创建 Cmai 的 `runBind`、`runEmbed` 运行环境，并按照 RoseTTAFold 官方说明完成其环境安装。安装完成后，请确认 Conda 环境中包含 `runEmbed`、`runBind` 和 `RoseTTAFold` 三个环境，并根据实际环境路径修改 `conf/env_path`，以及 `model/rfold/gen_embed.sh` 中对应的环境路径和环境名称。

### 权重与数据准备
Cmai 不仅依赖本模型包中的 Cmai 权重，还依赖 RoseTTAFold 权重以及 3 组 RoseTTAFold 序列/结构数据库。首次使用前请完成以下准备。

#### 1）Cmai 模型权重

通过 ModelScope 下载的模型包应已包含 Cmai 推理所需权重，统一放置在 `weight/` 目录。正常情况下无需单独下载 Cmai 权重；如目录内容缺失，请重新下载完整模型包。

```bash
modelscope download --model OneScience/Cmai --local_dir ./Cmai
cd Cmai
```

#### 2）安装 RoseTTAFold 源码

Cmai 会调用 RoseTTAFold 生成抗原结构相关 embedding，因此需要将 RoseTTAFold 放到 Cmai 的 `model/rfold/` 下：

```bash
cd /path/to/Cmai/model/rfold
git clone https://github.com/RosettaCommons/RoseTTAFold.git
cd RoseTTAFold
```

如已通过其他方式准备好 RoseTTAFold，可直接保证 `model/rfold/RoseTTAFold/` 指向可用源码目录。

#### 3）下载 RoseTTAFold 网络权重

在 RoseTTAFold 目录下执行：

```bash
wget https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz
tar xfz weights.tar.gz
```

解压后应能在 `model/rfold/RoseTTAFold/weights/` 中看到 RoseTTAFold 权重文件，例如：

```text
RoseTTAFold_e2e.pt
RoseTTAFold_pyrosetta.pt
RF2t.pt
```

#### 4）下载 RoseTTAFold 序列与结构数据库

建议准备一个独立数据库根目录，例如：

```text
/path/to/RoseTTAFold_database/
```

后续通过 `--rf_data /path/to/RoseTTAFold_database` 传给 Cmai。

**UniRef30_2020_06**

```bash
cd /path/to/RoseTTAFold_database
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06
```

**BFD**

```bash
cd /path/to/RoseTTAFold_database
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd
```

**PDB100_2021Mar03**

```bash
cd /path/to/RoseTTAFold_database
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz
```
- 下载并解压完成后，数据库目录通常类似：

```text
/path/to/RoseTTAFold_database/
├── UniRef30_2020_06/
├── bfd/
└── pdb100_2021Mar03/
```

- 上述数据库体积较大，UniRef30、BFD 和 PDB100 合计需要数百 GB 的磁盘存储空间，且解压后占用空间会进一步增加。下载前请确认目标文件系统具有足够的可用磁盘空间。

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/Cmai --local_dir ./Cmai
cd Cmai
```
- Cmai 额外依赖 RoseTTAFold；请先按照“权重与数据准备”完成 RoseTTAFold 源码、网络权重和 3 组数据库的准备。

### 快速验证

```bash
python scripts/Cmai.py --help
```

# 示例数据

`conf/data/` 目录保存示例输入、中间文件、背景数据和训练验证数据。

| 路径 | 说明 |
| --- | --- |
| `conf/data/example/input.csv` | 示例 BCR-抗原输入文件 |
| `conf/data/example/output/` | 示例输出目录 |
| `conf/data/intermediates/` | 预处理中间结果 |
| `conf/data/background/backgroundBCR.csv.gz` | 背景 BCR 数据 |
| `conf/data/background/default300.txt` | 默认背景抗原列表 |
| `conf/data/training_validation_data/` | 训练和验证数据 |

输入 CSV 至少需要包含以下列：

| 列名 | 说明 |
| --- | --- |
| `Antigen_id` | 抗原 ID |
| `BCR_Vh` | BCR 重链 V 区序列 |
| `BCR_CDR3h` | BCR 重链 CDR3 序列 |

如果输入文件中没有 `Antigen_seq` 列，则必须通过 `--fasta` 提供抗原 FASTA 文件。可选列包括 `BCR_id`、`Score`、`BCR_species` 等。包含 NA 的行会在预处理阶段被移除。

# 推理示例

## 一步式推理

一步式命令会自动完成预处理、Antigen embedding、`pair.npy` 整理和 binding prediction：

```bash
python scripts/Cmai.py \
  --code /path/to/Cmai \
  --input conf/data/example/input.csv \
  --out /path/to/Cmai/conf/data/example/output \
  --rf_data /path/to/RoseTTAFold_database \
  --Antigen_only
```
该命令会依次执行：

```text
输入检查与预处理
  → MSA 生成
  → RoseTTAFold 抗原 embedding
  → pair.npy 整理到 NPY/
  → Cmai binding prediction
  → 在背景 BCR 中计算 Antigen-anchor rank%
```
## 分阶段运行

本仓库同时提供了将 Antigen embedding 拆成一个步骤或两个步骤的方式。以下两个方案是**等价的两种执行方式**，不是需要全部顺序执行的三个独立阶段。

### 方案 A：一次完成 Antigen embedding

```bash
python scripts/Cmai.py \
  --code /path/to/Cmai \
  --input conf/data/example/input.csv \
  --out /path/to/Cmai/conf/data/example/output \
  --rf_data /path/to/RoseTTAFold_database \
  --runEmbed
```

`--runEmbed` 会完成 Antigen embedding，而不执行 binding prediction。

### 方案 B：将 Antigen embedding 拆成 MSA + RoseTTAFold 两步

**第 1 步：仅生成 MSA（CPU）**

```bash
python scripts/Cmai.py \
  --code /path/to/Cmai \
  --input conf/data/example/input.csv \
  --out /path/to/Cmai/conf/data/example/output \
  --rf_data /path/to/RoseTTAFold_database \
  --runEmbed \
  --gen_msa \
  --use_cpu
```

**第 2 步：复用已有 MSA，只运行 RoseTTAFold embedding**

```bash
python scripts/Cmai.py \
  --code /path/to/Cmai \
  --input conf/data/example/input.csv \
  --out /path/to/Cmai/conf/data/example/output \
  --rf_data /path/to/RoseTTAFold_database \
  --runEmbed \
  --run_rf
```

完成方案 A 或方案 B 后，再执行 binding prediction：

```bash
python scripts/Cmai.py \
  --code /path/to/Cmai \
  --out /path/to/Cmai/conf/data/example/output \
  --skip_check \
  --runBind \
  --Antigen_only
```

其中：

- `--skip_check`：跳过已经完成的输入检查/预处理，直接复用输出目录中的 `processed_input.csv` 等文件；
- `--runBind`：仅执行 binding prediction；
- `--Antigen_only`：固定输入抗原，在背景 BCR 中计算候选 BCR 的 rank%，不进入需要背景抗原 embedding 的 BCR-anchor 分支。

> **说明：** `--BCR_only` 模式以及未指定 `--Antigen_only` / `--BCR_only` 时的默认双向 `--runBind` 模式，需要额外的 background antigen embeddings。该数据未随仓库公开提供，如需使用 BCR-anchor ranking，请联系 Cmai 原作者申请相关背景抗原 embedding 数据。

## 仅生成或移动 embedding

如果 `RFoutputs/pred/` 中已有 `*.feature.npz`，需要单独提取其中的 `pair` 表征，可使用：

```bash
python scripts/Cmai.py \
  --code /path/to/Cmai \
  --out /path/to/Cmai/conf/data/example/output \
  --skip_check \
  --gen_npy
```

该参数通过主入口直接调用 `NPZtoPair.exPair()` 生成 `*.pair.npy`。

如果 `RFoutputs/pred/` 中已有 `*.pair.npy`，但尚未复制到预处理目录 `NPY/`，可使用：

```bash
python scripts/Cmai.py \
  --code /path/to/Cmai \
  --out /path/to/Cmai/conf/data/example/output \
  --skip_check \
  --move_npy
```

正常使用 `--runEmbed` 时主脚本会自动执行这一步，因此 `--gen_npy` / `--move_npy` 主要用于中断恢复、调试或手工复用已有 RoseTTAFold 结果。

# 输出说明

Antigen-only 示例成功运行后，输出目录通常包含：

```text
output/
├── antigens.fasta
├── processed_input.csv
├── RFoutputs/
├── NPY/
├── background_antigen_score_dict_10000BCRs.pkl
├── merged_results.csv
└── Skipped_entry.txt
```

其中：

- `RFoutputs/`：MSA、HHsearch、RoseTTAFold 预测及相关中间结果；
- `NPY/*.pair.npy`：Cmai binding 模型实际读取的抗原 pair embedding；
- `background_antigen_score_dict_10000BCRs.pkl`：Antigen-only 背景 BCR 排名过程中生成的评分字典；
- `merged_results.csv`：最终合并输入信息、结合分数与 rank 的结果文件。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证


- Cmai 原始论文：[Profiling antigen-binding affinity of B cell repertoires in tumors by deep learning predicts immune-checkpoint inhibitor treatment outcomes](https://doi.org/10.1038/s43018-025-01001-5)。

- Cmai 在抗原表征阶段依赖 RoseTTAFold。如果在研究工作中使用了 RoseTTAFold 生成抗原结构及相关表征，建议同时引用 RoseTTAFold 原始论文：[Accurate prediction of protein structures and interactions using a three-track neural network](https://www.science.org/doi/10.1126/science.abj8754)。

- Cmai 使用仓库提供的自定义学术研究许可证，详见仓库根目录 `LICENSE`。该许可证允许在满足许可证条款的前提下，以源代码或二进制形式使用、修改和再分发 Cmai，但**仅限学术研究用途（academic research use only）**。

- Cmai 许可证明确规定：**禁止任何商业用途或商业再分发**。无论是否修改代码，以源代码或二进制形式进行商业使用或再分发均被明确禁止；其中，**营利性实体（for-profit entity）对该软件的使用或再分发均视为商业用途**。因此，Cmai 软件不得用于商业用途或商业再分发。对于随模型包提供的模型权重、数据以及第三方资源，还应同时遵守其各自对应的发布许可和数据使用条款。

- 再分发 Cmai 源代码时，必须保留原始版权声明、许可证条件和免责声明；以二进制形式再分发时，也需要在相关文档或随附材料中保留上述信息。同时，未经书面许可，不得使用版权持有方或贡献者的名称为衍生产品进行背书或推广。该许可证不授予任何专利权，并按照 “AS IS” 原则提供软件和相关文档。

- RoseTTAFold 的**源代码**采用 MIT License，其**训练权重及相关数据**按照 Rosetta-DL Software License 提供，主要面向非商业用途。使用 RoseTTAFold 权重或相关数据时，请同时遵守其对应许可证要求，具体以 [RoseTTAFold 官方仓库](https://github.com/RosettaCommons/RoseTTAFold) 和 [Rosetta-DL Software License](https://files.ipd.uw.edu/pub/RoseTTAFold/Rosetta-DL_LICENSE.txt) 为准。

- 如果在科研工作中使用本仓库，建议同时引用 Cmai 原始论文和 OneScience 相关项目信息；若实际运行过程中使用了 RoseTTAFold 进行抗原 embedding 生成，还应补充 RoseTTAFold 原始论文引用。若进一步使用其他外部数据库、预训练模型或数据资源，也应按照相应资源的许可证和数据使用条款进行引用和使用。



