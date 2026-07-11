<h1 align="center">Evo2</h1>

## 模型介绍

Evo2 是面向基因组序列建模、生成和预测的大规模序列模型，可用于 DNA/RNA 序列生成、FASTA 序列预测、训练和微调流程验证。本目录是按单模型包形式整理的 Evo2 运行目录，包含本地模型源码、训练/推理脚本、示例数据、配置文件和权重放置目录。

当前目录支持的入口：

- Prompt 生成推理：`scripts/infer.py`
- FASTA 序列预测：`scripts/predict.py`
- 单节点训练或微调：`scripts/train.py`
- Slurm/集群训练：`scripts/train_slurm.py`
- FASTA/JSON 数据预处理：`scripts/tools/data_process/`

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `README.md` | 当前使用说明 |
| `configuration.json` | ModelScope/OneCode 元信息，声明训练、推理和预测入口 |
| `config/config.yaml` | 单模型运行配置，记录模型包路径、默认入口和权重目录 |
| `config/genome_data_config.yaml` | 训练数据集配置，配合 `--dataset-dir data/data_mini/genome_data` 使用 |
| `config/genome_preprocess_config.yaml` | genome 数据预处理配置 |
| `config/opengenome2.yml` | 额外训练配置文件，供平台或扩展训练流程使用 |
| `model/evo2/` | Evo2 本地 Python 包源码 |
| `scripts/infer.py` | prompt 生成推理入口 |
| `scripts/predict.py` | FASTA 预测入口 |
| `scripts/train.py` | 单节点训练入口 |
| `scripts/train_slurm.py` | Slurm/集群训练入口 |
| `scripts/tools/data_process/` | FASTA/JSON 数据预处理脚本 |
| `scripts/tools/checkpoint_convert/` | checkpoint 转换工具 |
| `data/predict_example.fa` | FASTA 预测示例输入 |
| `data/prompts.csv` | prompt 示例数据 |
| `data/data_mini/genome_data/` | 最小 genome 训练示例数据 |
| `checkpoints/` | 本地权重目录 |



## 使用说明

### 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://gitee.com/link?target=https%3A%2F%2Fweb-2069360198568017922-iaaj.ksai.scnet.cn%3A58043%2Fhome)

### 2. 手动安装使用

#### 硬件要求

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

#### 软件环境

建议在 Linux、GPU/DCU 和 OneScience BIO 环境中运行。本项目脚本依赖 `nemo`、`megatron.core`、`torch`、`lightning` 等组件，普通 Python 环境通常只能做源码查看，不能直接完成大模型训练或推理。

示例环境安装方式：

```bash
git clone https://gitee.com/onescience-ai/onescience.git
cd onescience
bash install.sh bio
```



### 3. 快速开始

在本目录运行时，脚本会自动把项目根目录和 `model/` 加入 `sys.path`。也可以手动验证本地包导入：

```bash
export PYTHONPATH=$(pwd)/model:${PYTHONPATH:-}
python -c "import evo2; print('evo2 import ok')"
```

#### 权重/数据下载

模型训练数据与权重数据需在魔塔社区按需下载。
确保现在位于evo2文件夹下：
```bash
cd onescience-examples/models/biosciences/evo2
```
数据放在 `data/` 的子目录中，下载前建议先创建 `data/data_mini`：

```bash
mkdir -p data/data_mini
modelscope download \
  --model OneScience/evo2 \
  --include 'data/**' \
  --local_dir .
```

权重放在 `checkpoints/` 的子目录中。当前 1B 训练默认从头开始，不需要下载 1B checkpoint；运行推理、预测或 7B 微调时再下载对应 checkpoint：

```bash
mkdir -p checkpoints
modelscope download \
  --model OneScience/evo2 \
  --include 'checkpoints/**' \
  --local_dir .
```

下载完成后，常用路径应为：

```text
data/data_mini/genome_data/
checkpoints/evo2_nemo_7b/
```

其中包含：

- `chr20.fa`、`chr21.fa`、`chr22.fa` 及压缩版本
- 合并 FASTA：`chr20_21_22.fa`
- 预处理后的训练、验证、测试二进制数据：`preprocessed_data/`

`config/genome_data_config.yaml` 中的 `dataset_prefix` 是相对路径，实际运行时需要配合：

```bash
--dataset-dir data/data_mini/genome_data
```



推理和预测入口会默认读取该目录；训练默认不加载 checkpoint。如需从已有权重微调或续训，请在训练命令中显式传入 `--ckpt-dir checkpoints/evo2_nemo_7b`。也可以通过 `EVO2_CKPT_DIR` 或 `--ckpt-dir` 覆盖为其他 checkpoint 路径：

```bash
export EVO2_CKPT_DIR=/path/to/evo2_nemo_7b
```

#### 快速检查

查看入口参数：

```bash
python scripts/infer.py --help
python scripts/predict.py --help
python scripts/train.py --help
python scripts/train_slurm.py --help
```

## Prompt 生成推理

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

## FASTA 预测

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

## 训练

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

## 数据预处理

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

## OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 引用与许可证

Evo2 原始实现包含 NVIDIA、Arc Institute、Stanford 等来源声明。本目录保留源码文件中的原始版权与许可证头信息，并面向 OneScience 单模型运行场景做了目录整理与入口适配。科研使用时，请同时引用 Evo2 原始论文、模型来源和 OneScience 项目信息。
