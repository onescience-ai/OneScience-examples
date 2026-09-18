<p align="center">
  <strong>
    <span style="font-size: 30px;">AlphaGenome</span>
  </strong>
</p>

# 模型介绍

AlphaGenome 是 Google DeepMind 提出的 DNA 序列模型，可输入最长 1 Mbp 的 DNA 区间并预测多类基因组功能信号，用于轨迹预测和调控变异效应评分。

论文：Advancing regulatory variant effect prediction with AlphaGenome  
https://www.nature.com/articles/s41586-025-10014-0

# 模型描述

AlphaGenome 基于 JAX / Flax 实现，支持基因组区间推理、变异效应评分、轨迹评估和微调示例。本模型包配套 ModelScope 数据集 `OneScience/alphagenome_dataset`，可用于本地快速验证。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 基因组区间预测 | 输入参考基因组 FASTA、染色体和区间坐标，输出 ATAC、DNase、CAGE、RNA-seq、ChIP 等预测轨迹 |
| 变异效应评分 | 输入 VCF 或内置示例变异，比较参考序列和变异序列预测差异，生成变异评分表 |
| 轨迹预测评估 | 使用 AlphaGenome 数据集中的验证数据，计算不同 assay bundle 的回归评估指标 |
| 微调实验 | 使用自定义参考基因组、区间 CSV 和 BigWig 信号文件开展微调流程验证 |
| ModelScope / OneCode 运行 | 下载模型工程和配套数据集后，在生物领域运行环境中快速验证脚本连通性 |



# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。





**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光 DCU：

```bash
hy-smi
```

### 下载模型包

```bash
modelscope download --model OneScience/alphagenome --local_dir ./alphagenome
cd alphagenome
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

安装完成后回到模型包目录：

```bash
cd ./alphagenome
```

### 训练与推理数据介绍

OneScience 社区已将 AlphaGenome 推理、评估和微调所需数据上传至 ModelScope：[OneScience/alphagenome_dataset](https://modelscope.cn/datasets/OneScience/alphagenome_dataset)。下载后请将数据放在模型包的 `data/` 目录。

```bash
modelscope download --dataset OneScience/alphagenome_dataset --local_dir ./data
```

### 训练权重

仓库已内置 `weight/alphagenome-all-folds`，同时脚本均支持通过 `--model_dir` 指定模型权重。

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

ModelScope 数据集 `OneScience/alphagenome_dataset` 建议下载到模型包下的 `data/`，默认结构如下：

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

- 本仓库基于 AlphaGenome 开源模型进行 DCU 适配，相关源码使用 Apache License 2.0。
- 科研使用请引用原始论文：[Advancing regulatory variant effect prediction with AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0)。
