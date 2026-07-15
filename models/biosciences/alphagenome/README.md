<p align="center">
  <strong>
    <span style="font-size: 30px;">AlphaGenome</span>
  </strong>
</p>

# 模型介绍

AlphaGenome 是 Google DeepMind 提出的统一 DNA 序列模型，可输入最长 1 Mbp 的 DNA 区间，并在碱基级或多尺度分辨率上预测多类基因组功能信号，覆盖 RNA-seq、CAGE、PRO-cap、ATAC-seq、DNase-seq、ChIP-seq、Hi-C / Micro-C 接触图谱、剪接位点与剪接连接等模态。模型可用于基因组轨迹预测和调控变异效应评分，适合非编码变异解释、功能基因组学分析和下游科研验证。

本模型包提供 AlphaGenome 的 JAX / Flax 模型实现，以及推理、变异评分、轨迹预测评估和微调示例脚本，可作为独立工程下载、部署和运行。

论文：Advancing regulatory variant effect prediction with AlphaGenome  
https://www.nature.com/articles/s41586-025-10014-0

# 仓库说明

本仓库是 AlphaGenome 最小可运行独立模型仓库，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 基因组区间多模态轨迹预测
- 单点变异效应评分
- track prediction 验证集评估
- 自定义基因组数据微调示例
- 本地 Orbax checkpoint 权重目录和共享运行目录两种路径约定

当前不支持能力：

- 不内置通用命令行平台或其它模型工程
- 不面向临床诊断或个人基因组医学判读

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 基因组区间预测 | 输入参考基因组 FASTA、染色体和区间坐标，输出 ATAC、DNase、CAGE、RNA-seq、ChIP 等预测轨迹 |
| 变异效应评分 | 输入 VCF 或内置示例变异，比较参考序列和变异序列预测差异，生成变异评分表 |
| 轨迹预测评估 | 使用 AlphaGenome 数据集中的验证数据，计算不同 assay bundle 的回归评估指标 |
| 微调实验 | 使用自定义参考基因组、区间 CSV 和 BigWig 信号文件开展微调流程验证 |
| OneCode / 本地运行 | 在生物领域运行环境中快速验证脚本连通性 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `flax_model/alphagenome/` | AlphaGenome 模型源码 | JAX / Flax 实现 |
| `flax_model/alphagenome/model/` | 模型结构、输出头、评分器和元数据 | 包含 interval / variant scoring 相关实现 |
| `flax_model/alphagenome/io/` | FASTA、数据集、基因组和剪接数据读取工具 | 供推理、评估和微调脚本调用 |
| `flax_model/alphagenome/finetuning/` | 微调数据集和训练步骤工具 | 用于 `scripts/run_finetuning.py` 示例 |
| `scripts/run_inference.py` | 区间推理脚本 | 支持本地 Orbax checkpoint；未指定权重时可走 Kaggle Hub 路径 |
| `scripts/run_variant_scoring.py` | 变异效应评分脚本 | 支持 VCF 输入或内置示例变异 |
| `scripts/run_track_prediction_eval.py` | track prediction 评估脚本 | 默认要求显式指定本地权重 |
| `scripts/run_finetuning.py` | 自定义数据微调示例脚本 | 需要 FASTA、区间 CSV 和 BigWig 输入 |
| `scripts/inference.sh` | 区间推理启动脚本 | 默认读取 `data/` 和 `weight/` |
| `scripts/run_variant.sh` | 变异评分启动脚本 | 默认读取 `data/` 和 `weight/` |
| `scripts/run_track.sh` | 轨迹评估启动脚本 | 默认读取 `data/` 和 `weight/` |
| `weight/` | 权重占位目录 | 建议放置 `weight/alphagenome-all-folds/` |
| `LICENSE` | 开源许可证 | Apache License 2.0 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**软件要求**

请参考 OneScience 生物领域运行环境，DCU 用户想了解更多适配内容请联系 liubiao@sugon.com。

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光 DCU：

```bash
hy-smi
```

## 3. 快速开始

### 安装运行环境

```bash
# 激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

安装完成后回到模型包目录：

```bash
cd ./alphagenome
```

### 准备权重

如使用本地权重，请将 AlphaGenome Orbax checkpoint 放置在以下目录：

```text
weight/
  alphagenome-all-folds/
    _CHECKPOINT_METADATA
    _METADATA
    ...
```

如果在共享运行环境中运行，也可以通过环境变量复用统一目录：

```bash
export ONESCIENCE_MODELS_DIR=/path/to/onescience/models
export ONESCIENCE_DATASETS_DIR=/path/to/onescience/datasets
```

脚本会优先读取：

- `${ONESCIENCE_MODELS_DIR}/AlphaGenome/alphagenome-all-folds`
- `${ONESCIENCE_DATASETS_DIR}/AlphaGenome`

未设置上述环境变量时，默认读取当前模型包下的：

- `weight/alphagenome-all-folds`
- `data/`

### 区间推理

```bash
bash scripts/inference.sh
```

等价的 Python 命令示例：

```bash
python scripts/run_inference.py \
  --fasta_path ./data/reference/HOMO_SAPIENS/GRCh38.p13.genome.fa \
  --model_dir ./weight/alphagenome-all-folds \
  --chromosome chr19 \
  --start 10587331 \
  --end 11635907 \
  --output_dir ./outputs
```

推理结果会保存至 `outputs/`。

### 变异效应评分

```bash
bash scripts/run_variant.sh
```

指定 VCF 输入时可使用：

```bash
python scripts/run_variant_scoring.py \
  --vcf_path ./data/example.vcf \
  --fasta_path ./data/reference/HOMO_SAPIENS/GRCh38.p13.genome.fa \
  --model_dir ./weight/alphagenome-all-folds \
  --output_dir ./outputs_variant
```

评分结果会保存为 CSV 文件。

### track prediction 评估

```bash
bash scripts/run_track.sh
```

也可以显式指定数据和输出路径：

```bash
python scripts/run_track_prediction_eval.py \
  --model_dir ./weight/alphagenome-all-folds \
  --model_version ALL_FOLDS \
  --data_dir ./data/v1/train \
  --output_path ./outputs_track/eval_results.csv
```

### 微调示例

```bash
python scripts/run_finetuning.py \
  --fasta_path ./data/reference/HOMO_SAPIENS/GRCh38.p13.genome.fa \
  --regions_csv ./data/finetune_regions.csv \
  --bigwig_paths ./data/sample_atac.bw \
  --output_dir ./finetuned_model \
  --num_steps 1000 \
  --batch_size 2
```

# 数据格式

请将所需参考基因组和评估数据准备到模型包下的 `data/`，默认结构如下：

```text
data/
  reference/
    HOMO_SAPIENS/
      GRCh38.p13.genome.fa
      GRCh38.p13.genome.fa.fai
  v1/
    train/
      ...
```

其中：

- `reference/HOMO_SAPIENS/GRCh38.p13.genome.fa` 为人类参考基因组 FASTA。
- `.fai` 为 FASTA 索引文件。
- `v1/train/` 为 track prediction 评估使用的数据目录。
- 自定义微调还需要准备区间 CSV 文件，列名为 `chromosome,start,end`，以及一个或多个 BigWig 信号文件。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- AlphaGenome 原始论文：Advancing regulatory variant effect prediction with AlphaGenome。
- 论文地址：https://www.nature.com/articles/s41586-025-10014-0
- AlphaGenome 相关源码使用 Apache License 2.0。
- 如果在科研工作中使用 AlphaGenome 结果，建议引用 AlphaGenome 原始论文和 OneScience 相关项目信息，并根据实际任务补充下游分析工具或数据集引用。
