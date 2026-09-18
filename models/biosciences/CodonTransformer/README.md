<p align="center">
  <strong>
    <span style="font-size: 30px;">CodonTransformer</span>
  </strong>
</p>

# 模型介绍

CodonTransformer 是一个面向多物种密码子优化的深度学习模型，用于根据输入蛋白质序列和目标宿主物种，生成宿主特异性的 DNA 编码序列。

由于遗传密码具有简并性，同一种蛋白质可以由大量不同 DNA 序列编码，而不同宿主对同义密码子的偏好并不相同。CodonTransformer 通过 Transformer 建模蛋白质、密码子和宿主物种之间的上下文关系，在尽量保持蛋白质翻译结果一致的前提下，生成更符合目标宿主天然密码子分布的 DNA 序列。

论文：
> **CodonTransformer: a multispecies codon optimizer using context-aware neural networks**  
> *Nature Communications*, 2025

# 模型描述

CodonTransformer 是一个面向多物种密码子优化任务的 Transformer 模型，核心网络采用 BigBird 掩码语言模型，并结合作者提出的 STREAM 表示方式，将目标宿主信息、氨基酸信息和密码子信息编码到统一的序列表征中。

与传统基于全局密码子频率的优化方法不同，CodonTransformer 不仅考虑单个密码子的宿主偏好，还利用 Transformer 的上下文建模能力学习相邻密码子之间的局部依赖关系，从而生成更接近目标宿主天然编码模式的 DNA 序列。模型可用于多物种密码子优化、异源蛋白表达序列设计，以及在自定义 DNA-蛋白质-宿主数据上的后续微调。

官方模型基于超过 100 万组 DNA-蛋白质配对样本训练，覆盖 164 个物种，包括细菌、古菌、植物、动物和真菌等。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| 密码子优化 | 根据目标宿主重新设计蛋白质编码 DNA |
| 异源蛋白表达 | 为不同宿主生成更符合宿主密码子偏好的编码序列 |
| 多物种序列设计 | 在多个支持宿主之间切换目标物种 |
| 多候选序列生成 | 通过温度采样生成多个不同 DNA 候选序列 |
| 批量密码子优化 | 对多个蛋白质和宿主组合进行批量推理 |
| 模型微调 | 使用自定义 DNA-蛋白质-宿主数据继续微调 |
| 模型预训练 | 使用大规模处理后的训练数据从头训练模型 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

### 硬件要求

- CodonTransformer 支持 CPU 和加速设备推理。
- 单条蛋白质序列推理可以使用 CPU。
- 批量推理、长序列推理、微调和预训练推荐使用 GPU/DCU。

### 环境安装

#### DCU/SCNet 环境

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311

pip install onescience[bio] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

- 在实际运行过程中，如遇到依赖缺失或版本不兼容等问题，可参考仓库根目录下的 `requirements.txt` 声明的依赖版本，补充安装或调整相应依赖。

### 模型与数据准备

#### 1）CodonTransformer 模型权重

官方模型发布在 Hugging Face：

```text
https://huggingface.co/adibvafa/CodonTransformer
```

联网环境下，from_pretrained 会在本地缓存不存在时自动下载模型权重；离线环境下必须提前下载到 HF_HOME 对应的缓存目录，建议缓存至：

```text
/path/to/.cache/huggingface/hub/
```

运行脚本时设置：

```bash
export HF_HOME=/path/to/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

仓库中的推理和微调脚本支持通过 `HF_HOME` 和 `OFFLINE=1` 使用本地缓存。

#### 2）训练数据集

官方训练数据可从 Zenodo 或 Hugging Face Dataset 获取：

```text
https://zenodo.org/records/12509224
https://huggingface.co/datasets/adibvafa/CodonTransformer
```

建议把下载后的原始数据统一放在：

```text
scripts/data/raw/
```

因此完整数据文件推荐保存为：

```text
scripts/data/raw/dataset.csv
```

处理后的训练 JSON 文件推荐保存到：

```text
scripts/data/processed/
```
## 3. 快速开始

### 下载模型包

```bash
modelscope download \
  --model OneScience/CodonTransformer \
  --local_dir ./CodonTransformer

cd CodonTransformer
```

# 推理示例

## 单序列密码子优化

执行脚本：

```bash
bash scripts/slurm/run_inference_single.sh
```

默认参数为：

```bash
PROTEIN="MFWY"
ORGANISM="Escherichia coli general"
OFFLINE=1
```

如需修改输入蛋白质和宿主：

```bash
PROTEIN="MALWMRLLPLLALLALWGPDPAAA" \
ORGANISM="Homo sapiens" \
bash scripts/slurm/run_inference_single.sh
```

## 生成多个候选 DNA 序列

可以直接运行：

```bash
bash scripts/slurm/run_inference_multiple.sh
```

默认会对同一条蛋白质序列生成多个候选 DNA 序列，并保存到：

```text
outputs/multiple_predictions.csv
```

核心参数包括：

```text
deterministic=False
temperature=0.5
top_p=0.95
num_sequences=5
match_protein=True
```

其中：

- `deterministic=False` 表示使用概率采样。
- `temperature` 控制采样多样性，常用范围为 `0.2 ~ 0.8`。
- `top_p` 控制 nucleus sampling。
- `num_sequences` 表示生成候选序列数量。
- `match_protein=True` 用于约束生成 DNA 翻译后仍与输入蛋白质一致。

## 更换目标宿主

目标宿主直接通过 `organism` 参数指定，例如：

```python
organism = "Escherichia coli general"
```

可以修改为：

```python
organism = "Homo sapiens"
```

或：

```python
organism = "Saccharomyces cerevisiae"
```

然后重新执行推理即可获得对应宿主的密码子优化 DNA。

## Batch 推理

示例数据当前位于：

```text
scripts/demo/sample_dataset.csv
```

直接执行：

```bash
bash scripts/slurm/run_inference_batch.sh
```

默认输出到：

```text
outputs/sample_predictions.csv
```

输入 CSV 至少需要包含：

```text
protein_sequence
organism
```

如果使用自定义 CSV：

```bash
INPUT_CSV=/path/to/input.csv \
OUTPUT_CSV=/path/to/output.csv \
bash scripts/slurm/run_inference_batch.sh
```

# 训练说明

## 微调数据准备

自定义微调首先准备用户自己的 CSV 文件，推荐放在：

```text
scripts/data/raw/your_data.csv
```

至少包含：

```text
dna
protein
organism
```

然后执行仓库中提供的数据预处理脚本：

```bash
INPUT_CSV=$PWD/scripts/data/raw/your_data.csv \
OUTPUT_JSON=$PWD/scripts/data/processed/finetune_data.json \
bash scripts/slurm/prepare_finetune_data.sh
```

## CodonTransformer 微调

执行脚本：

```bash
bash scripts/slurm/run_finetune.sh
```

默认读取：

```text
scripts/data/processed/finetune_data.json
```

默认保存到：

```text
weight/checkpoints/finetune
```

使用用户自定义数据进行微调时，可以执行：

```bash
DATASET_JSON=$PWD/scripts/data/processed/finetune_data.json \
CHECKPOINT_DIR=$PWD/weight/checkpoints/finetune \
CHECKPOINT_FILENAME=finetune.ckpt \
BATCH_SIZE=6 \
MAX_EPOCHS=15 \
NUM_WORKERS=5 \
ACCUMULATE_GRAD_BATCHES=1 \
NUM_GPUS=4 \
LEARNING_RATE=0.00005 \
WARMUP_FRACTION=0.1 \
SAVE_EVERY_N_STEPS=512 \
SEED=123 \
DEBUG=0 \
bash scripts/slurm/run_finetune.sh
```

其中 `NUM_GPUS=4`、`BATCH_SIZE=6`、`MAX_EPOCHS=15` 为默认训练设置。实际运行时应根据分配到的 GPU/DCU 数量、显存大小和数据规模调整这些参数。

## 导出微调模型并推理

微调完成后，可以使用以下脚本将 checkpoint 导出为推理可用的模型文件：

```text
scripts/slurm/export_finetuned_model.sh
```

执行方式如下：

```bash
CHECKPOINT_PATH=/path/to/finetuned_checkpoint.ckpt \
OUTPUT_MODEL_PATH=/path/to/output_finetuned_model.pt \
NUM_ORGANISMS=164 \
bash scripts/slurm/export_finetuned_model.sh
```

导出完成后，可以使用以下脚本加载微调模型进行推理：

```text
scripts/slurm/run_inference_finetuned.sh
```

执行方式如下：

```bash
PROTEIN="MFWY" \
ORGANISM="Escherichia coli general" \
MODEL_PATH=/path/to/output_finetuned_model.pt \
bash scripts/slurm/run_inference_finetuned.sh
```

## 预训练

当前预训练入口为：

```text
scripts/pretrain.py
```

预训练属于完整模型训练流程，需要准备大规模处理后的 DNA-蛋白质-宿主数据。完整预训练前，需要先把完整 `dataset.csv` 转换为训练脚本可读取的 JSONL 文件。

仓库提供了预处理脚本：

```text
scripts/slurm/prepare_pretrain_data.sh
```

该脚本默认读取：

```text
scripts/data/raw/dataset.csv
```

默认输出：

```text
scripts/data/processed/pretrain_data.json
```

因此，在完整预训练前应先执行：

```bash
bash scripts/slurm/prepare_pretrain_data.sh
```

如果原始数据或输出目录不是默认路径，可以通过环境变量修改：

```bash
INPUT_CSV=$PWD/scripts/data/raw/dataset.csv \
OUTPUT_JSON=$PWD/scripts/data/processed/pretrain_data.json \
bash scripts/slurm/prepare_pretrain_data.sh
```

生成的完整预训练数据建议保存为：

```text
scripts/data/processed/pretrain_data.json
```

完整预训练可以执行：

```bash
TRAIN_DATA_PATH=$PWD/scripts/data/processed/pretrain_data.json \
CHECKPOINT_DIR=$PWD/weight/checkpoints/pretrain \
BATCH_SIZE=6 \
MAX_EPOCHS=5 \
NUM_WORKERS=5 \
ACCUMULATE_GRAD_BATCHES=1 \
NUM_GPUS=16 \
LEARNING_RATE=0.00005 \
WARMUP_FRACTION=0.1 \
SAVE_INTERVAL=5 \
SEED=123 \
DEBUG=0 \
bash scripts/slurm/run_pretrain.sh
```

其中 `NUM_GPUS=16` 对应预训练脚本中的默认多卡设置。实际运行时应根据分配到的 GPU/DCU 数量调整 `NUM_GPUS`。预训练检查点默认保存到：

```text
weight/checkpoints/pretrain
```
# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |


# 引用与许可证

- CodonTransformer 官方源代码仓库采用 **Apache License 2.0**，详见仓库根目录`LICENSE`。
- CodonTransformer 模型权重通过 Hugging Face 独立发布，训练数据通过 Zenodo 和 Hugging Face Dataset 等渠道发布；模型权重、训练数据以及相关第三方资源的使用应分别遵守对应页面的许可证及使用条款。
- 本仓库为 CodonTransformer 的 **DCU适配版本**；仓库代码、模型权重及相关数据的使用仍应以各自原始项目中的许可证和使用条款为准。
