# Evo2

[OneScience 一键体验：Evo2](https://modelscope.cn/models/OneScience/evo2/)

## OneScience 官方信息

| 项目 | Gitee | GitHub |
|---|---|---|
| OneScience 文档 | https://gitee.com/onescience-ai/onescience-doc | https://github.com/onescience-ai/OneScience-doc |
| OneScience 主仓库 | https://gitee.com/onescience-ai/onescience | https://github.com/onescience-ai/OneScience |
| OneScience Skills | https://gitee.com/onescience-ai/oneskills | https://github.com/onescience-ai/oneskills |

## 项目说明

Evo2 是面向基因组序列建模的基础模型，基于 StripedHyena 2 / Hyena 架构，适合 DNA prompt 生成、FASTA 序列预测、基因组序列建模和基于 mini 数据的训练或微调冒烟测试。本仓库是 OneScience 的 ModelScope 标准模型仓库，包含 Evo2 7B NeMo2 权重、推理/训练脚本、mini 数据适配配置、Manifest 和预检脚本。

模型运行需要与数据集 `OneScience/evo2_dataset` 配合。数据集仓库提供 chr20、chr21、chr22 的 FASTA 与预处理 bin/idx split；模型仓库通过 `manifest.yaml` 的 `relations.required_datasets` 和 `run_matrix` 明确引用该数据集。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | model |
| ModelScope 模型 ID | `OneScience/evo2/` |
| 领域 | bio |
| 任务 | genome_sequence_modeling、prompt_generation、fasta_prediction、mini_train |
| 运行包类型 | `standard_runtime_package` |
| Manifest | `manifest.yaml` |
| 必需数据集 | `OneScience/evo2_dataset` |
| 默认工作目录 | 模型包根目录 |

## 文件说明

| 路径 | 类型 | 作用 | 必需 | 用途 | 默认放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明模型身份、文件、数据集关系、命令和输出 | 是 | 全部能力 | 模型包根目录 | 与 `onescience_run_manifest.yaml` 内容一致，修改后必须同步更新 |
| `README.md` | 说明文件 | 人类和大模型读取入口 | 是 | 全部能力 | 模型包根目录 | 正文以中文为主 |
| `checkpoints/evo2_nemo_7b/` | 模型权重 | Evo2 7B NeMo2 torch_dist checkpoint | 是 | 预检、推理、预测 | `checkpoints/evo2_nemo_7b/` | SHA256 在 `metadata/evo2_nemo_7b.sha256` |
| `infer.py` | 推理脚本 | DNA prompt 生成 | 是 | inference | 模型包根目录 | 需要 GPU 和 OneScience bio 依赖 |
| `predict.py` | 预测脚本 | 对 FASTA 做 logits/log probability 预测 | 是 | inference | 模型包根目录 | 需要数据集 FASTA |
| `train_one_node.py` | 训练脚本 | 单节点训练或微调 | 是 | train、finetune | 模型包根目录 | mini 命令用于冒烟测试 |
| `config/genome_data_config.yaml` | 数据配置 | 指向 chr20/21/22 预处理 train/validation/test split | 是 | preflight、train | `config/` | 与 `OneScience/evo2_dataset` 匹配 |
| `config/genome_preprocess_config.yaml` | 预处理配置 | 从 `chr20_21_22.fa` 生成 bin/idx | 是 | preprocess | `config/` | 数据集已包含预处理结果 |
| `config/opengenome2.yml` | 已适配配置 | 从全量 OpenGenome2 配置改为 mini 数据配置 | 是 | preflight、train | `config/` | 适配记录见 Manifest `configuration_adaptation` |
| `scripts/preflight_evo2.py` | 预检脚本 | 检查配置、权重、数据路径、文件大小和 SHA256 | 是 | preflight | `scripts/` | 可用 `--skip-sha256` 跳过大文件哈希 |
| `metadata/evo2_nemo_7b.*` | 校验清单 | 权重文件大小和 SHA256 | 是 | preflight | `metadata/` | 上传前和下载后校验使用 |

## Manifest

完整机器可读 Manifest 位于仓库根目录 `manifest.yaml`，同时保留同内容的兼容入口 `onescience_run_manifest.yaml` 以满足网页端读取习惯。大模型应先读取本 README 的 `## 文件说明` 和本节，再打开 `manifest.yaml` 或 `onescience_run_manifest.yaml` 并解析 YAML。修改运行脚本、权重路径、下载方式、数据集关系、命令或输出时，必须同步更新两个文件并重新运行 YAML 解析与预检。

## 模型 vs 数据集关系

本模型必须配合数据集 `OneScience/evo2_dataset`。模型 Manifest 中：

- `relations.required_datasets[0].resource_ref.repo_id` 为 `OneScience/evo2_dataset`。
- `run_matrix.scenarios` 中的 `minimal_validation`、`mini_train`、`fasta_prediction` 明确声明所需数据文件、命令和默认本地路径。
- 默认数据路径为模型包内 `data/evo2_dataset/data_mini/genome_data/`。

数据集仓库的 Manifest 通过 `relations.compatible_models` 反向声明兼容模型 `OneScience/evo2/`。

## 文件与下载

模型下载命令必须使用上传目标 ID：

```bash
modelscope download --model OneScience/evo2/
```

数据集下载命令必须使用数据集上传目标 ID：

```bash
modelscope download --dataset OneScience/evo2_dataset
```

如果网页端或用户使用 `modelscope download --cache_dir <DIR>`，下载完成后运行命令前必须切换到实际下载后的模型包根目录，也就是包含 `README.md` 和 `manifest.yaml` 的目录。不要在 OneScience 源码安装目录内直接改文件。

## 环境安装

如果当前会话已部署 OneScience bio 环境，可直接运行预检。环境缺失时，可在 OneScience 主仓库中安装 bio 依赖：

```bash
pip install -c constraints.txt .[bio]
```

推理、预测和训练需要 GPU、NeMo、Megatron、PyTorch 与 OneScience Evo2 依赖。预检脚本只需要 Python 和 PyYAML。

## 运行流程

### 1. 下载

```bash
modelscope download --model OneScience/evo2/
modelscope download --dataset OneScience/evo2_dataset
```

### 2. 放置数据

在模型包根目录执行：

```bash
mkdir -p data/evo2_dataset
cp -a <DATASET_REPO_ROOT>/data_mini data/evo2_dataset/data_mini
```

也可以不复制数据，在预检时用 `--data-root` 指向实际的 `genome_data` 目录。

### 3. 运行前预检

```bash
python scripts/preflight_evo2.py --package-root . --data-root data/evo2_dataset/data_mini/genome_data
```

如只想快速检查结构，可临时跳过权重大文件哈希：

```bash
python scripts/preflight_evo2.py --package-root . --data-root data/evo2_dataset/data_mini/genome_data --skip-sha256
```

### 4. Prompt 推理

```bash
python infer.py --ckpt-dir checkpoints/evo2_nemo_7b --prompt ATGCGT --max-new-tokens 32 --output-file outputs/evo2_prompt.txt
```

### 5. FASTA 预测

```bash
python predict.py --fasta data/evo2_dataset/data_mini/genome_data/chr22.fa --ckpt-dir checkpoints/evo2_nemo_7b --model-size 7b --output-dir outputs/evo2_predict
```

### 6. Mini 训练冒烟测试

```bash
python train_one_node.py -d config/genome_data_config.yaml --dataset-dir data/evo2_dataset/data_mini/genome_data --model-size 1b --devices 1 --num-nodes 1 --seq-length 8192 --micro-batch-size 1 --lr 0.0001 --warmup-steps 5 --max-steps 10 --clip-grad 1 --wd 0.01 --val-check-interval 5
```

## 输出说明

| 输出 | 来源命令 | 说明 |
|---|---|---|
| 终端 `[OK] Evo2 model package...` | `scripts/preflight_evo2.py` | 模型、配置、权重和数据路径预检通过 |
| `outputs/evo2_prompt.txt` | `infer.py` | Prompt 生成结果 |
| `outputs/evo2_predict/` | `predict.py` | FASTA 预测结果目录 |
| Lightning 日志或 checkpoint | `train_one_node.py` | 训练输出，具体路径受 NeMo logger 配置影响 |

## 预检与诊断

| 错误现象 | 可能原因 | 处理方式 |
|---|---|---|
| `missing dataset file` | 未下载或未放置 `OneScience/evo2_dataset` | 下载数据集并将 `data_mini` 放到 `data/evo2_dataset/data_mini` |
| `sha256 mismatch` | 权重文件复制或下载损坏 | 重新下载 `OneScience/evo2/` |
| `ModuleNotFoundError: nemo` | OneScience bio 环境未激活 | 激活或安装 OneScience bio 依赖 |
| `torch.cuda.device_count` 不足 | GPU 不可用或并行参数过大 | 换 GPU 节点或降低并行参数 |
| 下载命令指向其他 ID | README 或 Manifest 被改错 | 模型必须使用 `OneScience/evo2/`，数据集必须使用 `OneScience/evo2_dataset` |

## 限制与适用范围

本仓库适合 Evo2 7B 的 OneScience 标准预检、prompt 推理、FASTA 预测和 mini 数据训练冒烟测试。完整 OpenGenome2 训练需要全量数据与更大规模算力，不由本 mini 数据集覆盖。`config/opengenome2.yml` 已从全量配置适配为 chr20/21/22 mini 数据；原始模型目录和原始数据目录未被修改。

## 引用与许可证

Evo2 论文：Genome modeling and design across all domains of life with Evo 2。代码许可继承 OneScience/Evo2 上游声明；权重和数据许可请遵循对应来源条款。
