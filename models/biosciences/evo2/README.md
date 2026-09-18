<p align="center">
  <strong><span style="font-size: 30px;">Evo2</span></strong>
</p>

# 模型介绍

Evo2 是一个面向基因组序列的大规模基础模型，能够对 DNA/RNA 序列进行生成、补全和功能预测。它支持在大规模基因组数据集上预训练，并可快速适配到调控元件预测、变异效应评估等下游任务，为基因组学研究提供了统一的序列建模工具。

# 模型描述

Evo2 基于 Hyena 算子与卷积门控混合架构，支持百万级上下文窗口。模型在数万亿碱基对的多物种基因组数据上预训练。输入原始核苷酸序列，支持自回归生成、序列分类与嵌入提取。兼容多 GPU 分布式训练与灵活数据混合，在 GenBench 等基准中性能领先，适用于预训练、微调及下游功能验证。

# 适用场景

| 场景 | 说明 |
| --- | --- |
| DNA/RNA 序列生成 | 输入 prompt 后生成后续基因组序列片段 |
| FASTA 序列预测 | 读取 FASTA 输入并输出序列预测结果 |
| 本地离线推理验证 | 使用仓库内示例输入和本地 checkpoint 跑通推理链路 |
| mini 训练流程验证 | 使用 `data/data_mini/genome_data/` 验证训练脚本、数据读取和 checkpoint 流程 |
| 集群训练适配 | 使用 `scripts/train_slurm.py` 在 Slurm/集群环境中运行训练 |
| 数据预处理 | 对 FASTA/JSON 数据生成 Byte-Level tokenizer 所需的 `.bin/.idx` 数据 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://gitee.com/link?target=https%3A%2F%2Fweb-2069360198568017922-iaaj.ksai.scnet.cn%3A58043%2Fhome)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/evo2 --local_dir ./evo2
cd evo2
```

### 安装运行环境
#### DCU环境

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

如果脚本需要访问 OneScience 源码，可设置：

```bash
export ONESCIENCE_ROOT=/path/to/onescience
```

在本目录运行时，脚本会自动把项目根目录和 `model/` 加入 `sys.path`。也可以手动验证本地包导入：

```bash
export PYTHONPATH=$(pwd)/model:${PYTHONPATH:-}
python -c "import evo2; print('evo2 import ok')"
```

## 3. 快速开始

### 快速检查

查看入口参数：

```bash
python scripts/infer.py --help
python scripts/predict.py --help
python scripts/train.py --help
python scripts/train_slurm.py --help
```

## 4. 数据与权重

示例训练数据已放在：

```text
data/data_mini/genome_data/
```

其中包含：

* `chr20.fa`、`chr21.fa`、`chr22.fa` 及压缩版本
* 合并 FASTA：`chr20_21_22.fa`
* 预处理后的训练、验证、测试二进制数据：`preprocessed_data/`

`config/genome_data_config.yaml` 中的 `dataset_prefix` 是相对路径，实际运行时需要配合：

```bash
--dataset-dir data/data_mini/genome_data
```

默认 7B NeMo checkpoint 放置在仓库内：

```text
checkpoints/evo2_nemo_7b/
```

推理和预测入口会默认读取该目录；训练默认不加载 checkpoint。如需从已有权重微调或续训，请在训练命令中显式传入 `--ckpt-dir checkpoints/evo2_nemo_7b`。也可以通过 `EVO2_CKPT_DIR` 或 `--ckpt-dir` 覆盖为其他 checkpoint 路径：

```bash
export EVO2_CKPT_DIR=/path/to/evo2_nemo_7b
```

权重文件在/path/to/evo2/checkpoints/evo2_nemo_7b/weights路径下

## 5. Prompt 生成推理

`infer.py` 会把结果写入 `--output-file`，但不会自动创建父目录。首次运行前先创建输出目录：

```bash
mkdir -p outputs
```

最小示例：

```bash
python scripts/infer.py \
  --prompt "ATGCGT" \
  --output-file outputs/evo2_generation.txt
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--ckpt-dir` | `checkpoints/evo2_nemo_7b` | NeMo checkpoint 目录，可用 `EVO2_CKPT_DIR` 覆盖 |
| `--prompt` | 内置 E. coli 分类标签 prompt | 输入序列或文本 prompt |
| `--max-new-tokens` | `1024` | 最大生成 token 数 |
| `--temperature` | `1.0` | 采样温度 |
| `--top-k` | `0` | top-k 采样 |
| `--top-p` | `0.0` | top-p 采样 |
| `--tensor-parallel-size` | `1` | 张量并行规模 |
| `--output-file` | 空 | 指定时写入文件，否则打印日志 |

## 6. FASTA 预测

目录内置示例 FASTA：

```text
data/predict_example.fa
```

运行预测：

```bash
python scripts/predict.py \
  --fasta data/predict_example.fa \
  --output-dir outputs/predict
```

`predict.py` 会自动创建 `--output-dir`。如不传 `--fasta`，默认读取 `data/predict_example.fa`；如不传 `--output-dir`，默认输出到 `outputs/predict`。

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--fasta` | `data/predict_example.fa` | 输入 FASTA 文件 |
| `--ckpt-dir` | `checkpoints/evo2_nemo_7b` | NeMo checkpoint 目录，可用 `EVO2_CKPT_DIR` 覆盖 |
| `--output-dir` | `outputs/predict` | 预测结果目录 |
| `--batch-size` | `1` | 预测 batch size |
| `--model-size` | `7b` | 模型规格，可选值由 NeMo `HYENA_MODEL_OPTIONS` 决定 |
| `--output-log-prob-seqs` | 关闭 | 输出序列 log probability |

`data/data_mini/genome_data/chr20.fa`、`chr21.fa`、`chr22.fa` 是较大的染色体级 FASTA。直接预测可能占用较高显存，建议先切分为较短片段，或使用多卡并行配置。

## 7. 训练

训练入口要求二选一：传入 `-d/--dataset-config`，或使用 `--mock-data`。内置示例数据推荐使用 `config/genome_data_config.yaml` 和 `data/data_mini/genome_data`。

使用 1B 架构从头训练或做训练流程验证示例：

```bash
python scripts/train.py \
  -d config/genome_data_config.yaml \
  --dataset-dir data/data_mini/genome_data \
  --model-size 1b \
  --result-dir results_1b \
  --devices 8 \
  --num-nodes 1 \
  --seq-length 8192 \
  --micro-batch-size 2 \
  --lr 0.0001 \
  --warmup-steps 5 \
  --max-steps 1000 \
  --clip-grad 1 \
  --wd 0.01 \
  --activation-checkpoint-recompute-num-layers 1 \
  --val-check-interval 50 \
  --limit-val-batches 2
```

该示例不加载 checkpoint，适合在没有 1B 权重时验证训练链路。如需从 1B NeMo checkpoint 微调，可额外传入 `--ckpt-dir /path/to/evo2_nemo_1b`。

使用 7B 长上下文架构从头训练或做训练流程验证示例：

```bash
python scripts/train.py \
  -d config/genome_data_config.yaml \
  --dataset-dir data/data_mini/genome_data \
  --model-size 7b_arc_longcontext \
  --result-dir results_7b \
  --devices 8 \
  --num-nodes 1 \
  --seq-length 1024 \
  --micro-batch-size 1 \
  --lr 0.0001 \
  --warmup-steps 5 \
  --max-steps 1000 \
  --clip-grad 1 \
  --wd 0.01 \
  --activation-checkpoint-recompute-num-layers 1 \
  --val-check-interval 50 \
  --limit-val-batches 2
```

该示例不加载 checkpoint。若需要从已有 7B NeMo checkpoint 微调或续训，可额外传入 `--ckpt-dir checkpoints/evo2_nemo_7b` 或其他 checkpoint 路径。

Slurm/集群环境可把入口替换为：

```bash
python scripts/train_slurm.py ...
```

常用训练参数：

下表覆盖上面训练示例和 `scripts/train_evo2_1b.sh`、`scripts/train_evo2_7b.sh` 中用到的主要参数；更多高级参数可运行 `python scripts/train.py --help` 查看。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-d, --dataset-config` | 必填或使用 `--mock-data` | 训练数据配置 YAML |
| `--mock-data` | 关闭 | 不读取真实数据，使用 mock 数据做连通性测试；与 `-d/--dataset-config` 二选一 |
| `--dataset-dir` | `EVO2_DATASET_DIR` 或 `data/data_mini/genome_data` | 数据根目录 |
| `--ckpt-dir` | 空 | 初始 checkpoint 目录；从已有权重微调或续训时显式传入 |
| `--model-size` | `7b` | 模型规格，常用 `1b`、`7b`、`7b_arc_longcontext`，测试可用 `test` |
| `--devices` | `1` | 单节点设备数 |
| `--num-nodes` | `1` | 节点数 |
| `--tensor-parallel-size` | `1` | 张量并行规模 |
| `--pipeline-model-parallel-size` | `1` | 流水线并行规模 |
| `--context-parallel-size` | `1` | context parallel 规模 |
| `--sequence-parallel` | 关闭 | 启用 sequence parallel，通常配合张量并行使用 |
| `--seq-length` | `8192` | 训练序列长度 |
| `--micro-batch-size` | `1` | micro batch size |
| `--global-batch-size` | 自动推断 | global batch size |
| `--grad-acc-batches` | `1` | 梯度累积 batch 数，用于推断 global batch size |
| `--lr` | `3e-4` | 学习率，示例中显式设为 `0.0001` |
| `--min-lr` | `3e-5` | cosine scheduler 最小学习率 |
| `--warmup-steps` | `2500` | warmup 步数，示例中显式设为 `5` |
| `--constant-steps` | `80000` | 学习率保持常量的步数 |
| `--max-steps` | `500000` | 训练步数 |
| `--wd` | `0.01` | optimizer weight decay |
| `--clip-grad` | `1.0` | 梯度裁剪阈值 |
| `--workers` | `8` | DataLoader worker 数 |
| `--activation-checkpoint-recompute-num-layers` | 空 | 覆盖激活重计算层数，示例中显式设为 `1` |
| `--val-check-interval` | 空 | 验证和 checkpoint 检查间隔，示例中显式设为 `50` |
| `--limit-val-batches` | `20` | 每次验证最多运行的 batch 数，示例中显式设为 `2` |
| `--result-dir` | `./results` | 日志与结果目录 |
| `--experiment-name` | `evo2` | 实验名称，影响日志目录 |
| `--disable-checkpointing` | 默认不传，checkpoint 回调启用 | 连通性测试时可传入该参数关闭 checkpoint 回调 |
| `--no-save-last-checkpoint` | 默认保存 last checkpoint | 禁止保存最后一个 checkpoint，常与 `--disable-checkpointing` 一起用于快速测试 |
| `--save-top-k` | `5` | 保存最优 checkpoint 数量 |
| `--ckpt-async-save` | 关闭 | 启用异步 checkpoint 保存 |

## 8. 数据预处理

当前示例训练数据已预处理。如需处理新的 FASTA 或 JSON 数据，可使用：

```bash
bash scripts/tools/data_process/preprocess_data_fasta.sh
bash scripts/tools/data_process/preprocess_data_json.sh
```

相关配置：

```text
config/genome_preprocess_config.yaml
config/genome_data_config.yaml
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Evo2 原始论文：[Genome modelling and design across all domains of life with Evo 2](https://doi.org/10.1038/s41586-026-10176-5)。
- Evo2 原始代码使用 Apache License 2.0，见本仓库 `LICENSE`。原始许可证来源：[ArcInstitute/evo2 LICENSE](https://github.com/ArcInstitute/evo2/blob/main/LICENSE)。
- Evo2 7B 模型权重来源于 HuggingFace [`arcinstitute/evo2_7b`](https://huggingface.co/arcinstitute/evo2_7b)，本仓库中 `checkpoints/evo2_nemo_7b` 为转换后的 NeMo checkpoint 格式。原始权重许可证为 Apache-2.0。
- 示例数据集 `data/data_mini/genome_data/` 由 UCSC Genome Browser 提供的 hg38 `chr20`、`chr21`、`chr22` chromosome FASTA 下载并预处理得到，仅用于训练和推理流程验证。数据下载目录见：[UCSC hg38 chromosomes](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/)。
- UCSC hg38 数据的使用请遵守 UCSC Genome Browser 数据使用条款，见：[UCSC 使用条款](https://genome.ucsc.edu/license/)。
- 如果在科研工作中使用本仓库、模型权重或生成结果，建议引用 Evo2 原始论文、模型权重来源、UCSC hg38 数据来源和 OneScience 相关项目信息，并根据实际任务补充下游分析工具或数据集引用。

