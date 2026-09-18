<p align="center">
  <strong>
    <span style="font-size: 30px;">Geneformer</span>
  </strong>
</p>

# 模型介绍

Geneformer 是由哈佛大学/波士顿儿童医院 Christina Theodoris 团队研发的单细胞转录组基础模型。模型在 Genecorpus-30M（约 3000 万个人类单细胞）上以掩码语言模型目标预训练，通过 rank value encoding 将基因表达转换为上下文感知的 Token 序列，支持零样本预测与轻量微调，在剂量敏感转录因子预测、染色质动力学、心肌病分类、in-silico 扰动等网络生物学任务上取得领先结果。

论文：Transfer learning enables predictions in network biology

https://www.nature.com/articles/s41586-023-06139-9



# 模型描述

Geneformer 包含 V1（6 层 / 256 隐藏维，Genecorpus-10M）与 V2（12 层 / 512 隐藏维，104M 与 316M）两代模型，基础模型采用 Hugging Face BERT 结构；官方自定义的多任务分类网络（`GeneformerMultiTask`）位于本仓库 `model/` 目录，为自包含实现。数据管线（分词、基因字典、collator）、分类器、嵌入提取、预训练器与扰动工具通过环境中已安装的 onescience 包提供。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 细胞分类微调 | 在带标签的 tokenized 数据上微调细胞状态分类器（如心肌病疾病分类） |
| 基因分类微调 | 剂量敏感转录因子等基因级分类，支持分层交叉验证与全量训练 |
| 嵌入推理 | 提取 CLS / cell / gene 嵌入，输出 CSV 或张量 |
| 转录组分词 | 将 raw counts 的 Loom / H5AD / Zarr 数据分词为 `.dataset` |
| 预训练 | 以官方 V1 配置在 tokenized 语料上执行掩码语言模型预训练 |
| In-silico 扰动 | 基因删除 / 过表达的零样本扰动预测 |
| 多任务微调 | 单机或分布式多任务分类 |
| 本地快速验证 | 使用少量细胞（`--max-cells`）快速验证完整链路 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入、分词和小配置连通性验证，训练和推理速度较慢。
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
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

```bash
# 如果需要找不到库的情况需要激活cuda，参考下列代码
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"
```

安装完成后回到模型包目录：

```bash
cd ./Geneformer
```

### 下载数据集&权重

**权重**

从 ModelScope 模型仓库 `OneScience/Geneformer` 下载 `weight/**` 到 `weight/`：

```bash
modelscope download --model OneScience/Geneformer --include "weight/**" --local_dir ./Geneformer
```

包含以下预训练与微调模型：

```text
weight/
  Geneformer-V1-10M/                                              # V1 预训练模型（10M）
  Geneformer-V2-104M/                                             # V2 预训练模型（104M）
  Geneformer-V2-104M_CLcancer/                                    # V2 CL 癌症分类模型
  Geneformer-V2-316M/                                             # V2 预训练模型（316M）
  fine_tuned_models/
    Geneformer-V1-10M_CellClassifier_cardiomyopathies_220224/     # V1 心肌病细胞分类模型
  geneformer/                                                     # 基因字典（gc30M/gc104M）
```

每个模型目录需包含 `config.json` 与 `model.safetensors`（或 `pytorch_model.bin`）。

**数据**

演示数据从 ModelScope 数据集仓库 `OneScience/Geneformer_dataset` 下载到 `data/`：

```bash
modelscope download --dataset OneScience/Geneformer_dataset --local_dir ./Geneformer/data
```

默认路径 `data/Genecorpus-30M/...` 与下表环境变量默认值一致。

| 环境变量 | 默认值 | 说明 |
| :---: | :---: | :---: |
| `GENEFORMER_MODEL_ROOT` | `weight` | 权重根目录 |
| `GENEFORMER_DATASET_ROOT` | `data` | 数据集根目录 |
| `GENEFORMER_V1_MODEL` | `weight/Geneformer-V1-10M` | V1 预训练模型目录 |
| `GENEFORMER_V1_CELL_MODEL` | `weight/fine_tuned_models/Geneformer-V1-10M_CellClassifier_cardiomyopathies_220224` | V1 心肌病细胞分类模型目录 |
| `GENEFORMER_V1_CORPUS` | `data/Genecorpus-30M/genecorpus_30M_2048.dataset` | V1 预训练语料 |
| `GENEFORMER_V1_LENGTHS` | `data/Genecorpus-30M/genecorpus_30M_2048_lengths.pkl` | V1 语料长度 |
| `GENEFORMER_CELL_DATA` | `data/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset` | 细胞分类数据 |
| `GENEFORMER_GENE_DATA` | `data/Genecorpus-30M/example_input_files/gene_classification/dosage_sensitive_tfs/gc-30M_sample50k.dataset` | 基因分类数据 |
| `GENEFORMER_GENE_CLASSES` | 同目录 `dosage_sensitivity_TFs.pickle` | 基因类别字典 |
| `GENEFORMER_OUTPUT_ROOT` | `outputs` | 输出根目录 |

### 嵌入推理

```bash
bash scripts/inference.sh --max-cells 64 --batch-size 16
```

结果保存为 `outputs/embeddings/cardiomyopathy_cell_embeddings.csv`。也可直接调用 `extract_embeddings.py` 选择预训练模型、V2 模型、CLS/cell/gene 嵌入或其他标签列。

### 细胞分类微调

默认按 `disease` 列微调 V1 模型，并保留独立测试集：

```bash
bash scripts/finetune.sh \
  --max-cells 1000 \
  --epochs 1 \
  --batch-size 12
```

仅验证数据准备时可增加 `--prepare-only`。官方明确建议针对具体任务进行超参数搜索，脚本提供的参数只用于演示流程：

```bash
bash scripts/finetune.sh --hyperparameter-trials 10
```

启用该可选搜索路径前需安装 `hyperopt`。

### 基因分类微调

默认使用 V1 dosage-sensitive transcription-factor 数据与类别字典执行分层交叉验证：

```bash
bash scripts/finetune_gene_classifier.sh \
  --max-cells 1000 \
  --cross-validation-splits 1 \
  --epochs 1
```

`--train-all-data` 对应官方在全部标注数据上训练最终模型的流程；`--gene-balance` 可用于官方支持的二分类基因平衡。

### 预训练

默认配置与官方 V1（6 层、256 隐藏维）一致：

```bash
bash scripts/pretrain.sh
```

完整 Genecorpus-30M 训练开销很大。冒烟测试可限制样本和步数：

```bash
bash scripts/pretrain.sh \
  --max-cells 64 \
  --max-steps 1 \
  --batch-size 2 \
  --overwrite-output-dir
```

### 转录组分词

输入必须是未经 feature selection 的 raw counts，并提供 Ensembl ID；H5AD/Zarr 默认从 `var["ensembl_id"]` 读取，也可通过 `--use-h5ad-index` 使用 `var_names`：

```bash
python scripts/tokenize_transcriptomes.py \
  --input-dir /path/to/raw_h5ad_directory \
  --output-dir outputs/tokenized \
  --output-prefix cells_v2 \
  --file-format h5ad \
  --model-version V2 \
  --metadata cell_type=cell_type
```

### In-silico perturbation

不指定基因时会逐细胞测试所有检测到的基因，开销可能很大；建议先指定少量 Ensembl ID：

```bash
bash scripts/perturb.sh \
  --max-cells 2 \
  --gene ENSG00000141510
```

原始扰动结果可继续交给 `onescience.utils.geneformer.InSilicoPerturberStats` 执行官方支持的 goal-state、null distribution、mixture model 或聚合统计。

### 多任务分类微调

输入必须是已经切分的 tokenized Dataset，并包含 `unique_cell_id` 以及每个任务的标签列：

```bash
python scripts/multitask_finetune.py \
  --model-dir "${GENEFORMER_V1_MODEL}" \
  --train-data /path/to/train.dataset \
  --validation-data /path/to/validation.dataset \
  --task-column disease \
  --task-column cell_type
```

# 数据格式

Geneformer 主要使用以下格式：

```text
*.loom / *.h5ad / *.zarr   原始计数转录组（分词输入，需 raw counts 与 Ensembl ID）
*.dataset                  HF Dataset 格式的分词数据（模型训练与推理输入）
```

演示数据由 ModelScope 数据集仓库 `OneScience/Geneformer_dataset` 提供（Genecorpus-30M 与 Genecorpus-104M，含示例切分与基因字典），模型权重由模型仓库 `OneScience/Geneformer` 的 `weight/` 目录提供。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Geneformer 原始论文：Transfer learning enables predictions in network biology。
- 论文地址：https://www.nature.com/articles/s41586-023-06139-9
- 官方代码与模型：https://huggingface.co/ctheodoris/Geneformer
- 官方数据：https://huggingface.co/datasets/ctheodoris/Genecorpus-30M
- Geneformer 官方代码使用 Apache-2.0 许可证（详见 `model/LICENSE.geneformer`）。
- 如果在科研工作中使用 Geneformer 结果，建议引用：

```bibtex
@article{theodoris2023transfer,
  title   = {Transfer learning enables predictions in network biology},
  author  = {Theodoris, Christina V. and Xiao, Ling and Chopra, Anant and
             Chaffin, Mark D. and Al Sayed, Zeina R. and Hill, Matthew C. and
             Mantineo, Helene and Brydon, Elizabeth M. and Zeng, Zexian and
             Liu, X. Shirley and Ellinor, Patrick T.},
  journal = {Nature},
  year    = {2023},
  doi     = {10.1038/s41586-023-06139-9}
}
```


