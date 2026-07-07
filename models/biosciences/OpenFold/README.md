---
frameworks:
- ""
---
# OpenFold

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 OpenFold 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/OpenFold" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

OpenFold 是面向蛋白质三维结构预测的深度学习模型实现，使用蛋白质 FASTA 序列、MSA 对齐结果和模板结构信息预测单体或相关预设下的结构坐标。它适合在生物信息场景中进行结构预测、推理流程验证、模型微调入口验证，以及基于用户自备序列和外部数据库的结构建模。

本仓库是 OneScience 标准运行包形式的 OpenFold 模型资源，上传内容不是孤立权重，而是来自 `examples/biosciences/openfold` 的代码、脚本、示例 FASTA、训练入口、推理入口、DeepSpeed 配置和 `params/finetuning_ptm_2.pt` 权重。网页端大模型下载本仓库后，应以仓库根目录作为运行工作目录，并优先读取 `manifest.yaml` 生成下载、预检和运行计划。

OpenFold 的 MSA、模板 mmCIF、FASTA 和训练数据通常由用户提供，或由用户在 OneScience 环境中准备 AlphaFold/OpenFold 外部数据库后生成。本模型包不绑定固定 ModelScope 数据集仓库，因此 Manifest 中 `relations.required_datasets` 为空；运行时仍必须按命令参数提供必要输入路径。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| ModelScope ID | `OneScience/OpenFold` |
| OneScience 领域 | `bio` |
| 领域标签 | `bio`, `protein`, `protein_structure_prediction` |
| 任务 | 蛋白质结构预测 |
| 任务标签 | `protein_structure_prediction`, `folding`, `ptm_prediction` |
| 主平台资源 | https://modelscope.cn/models/OneScience/OpenFold |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/biosciences/openfold` |
| 支持能力 | 预检、推理入口、训练入口、辅助脚本 |
| 必需模型文件 | `params/finetuning_ptm_2.pt` |
| 必需数据集 | 无固定 ModelScope 数据集依赖；FASTA、MSA、模板和训练数据由用户提供或外部数据库准备 |
| 最小验证 | `python tools/preflight_openfold.py --repo-root .` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 说明文件 | 面向人类和大模型的运行入口 | 是 | 全部能力 | 仓库根目录 | 读取后继续解析 `manifest.yaml` |
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、关系、命令和输出 | 是 | 全部能力 | 仓库根目录 | 修改运行包后必须同步更新 |
| `onescience_run_manifest.yaml` | Manifest 副本 | 兼容 OneScience 网页端对运行清单文件名的读取 | 是 | 全部能力 | 仓库根目录 | 内容与 `manifest.yaml` 保持一致 |
| `onescience_relations.yaml` | 关系文件 | 显式声明本模型无固定必需数据集仓库，输入由用户或外部数据库准备 | 是 | 解析关系 | 仓库根目录 | 与 Manifest 的 `relations` 保持一致 |
| `run_pretrained_openfold.py` | 推理脚本 | 使用 FASTA、模板 mmCIF、MSA 和 checkpoint 执行结构预测 | 是 | 推理 | 仓库根目录 | OneScience OpenFold 推理入口 |
| `train_openfold.py` | 训练脚本 | 使用 mmCIF、预计算 alignment、模板和训练配置执行训练 | 训练时必需 | 训练 | 仓库根目录 | 训练数据由用户提供 |
| `monomer/inference.sh` | 示例脚本 | 单体推理命令示例，展示 FASTA、模板、alignment 和 checkpoint 参数 | 否 | 推理参考 | `monomer/` | 默认路径依赖 OneScience 环境变量，可按实际数据修改 |
| `monomer/fasta_dir/6kwc.fasta` | 示例输入 | 最小目录结构验证使用的 FASTA 文件 | 是 | 预检、推理准备 | `monomer/fasta_dir/` | 完整推理仍需要 alignment 和模板数据 |
| `monomer/alignments/` | 输入目录 | 预计算 MSA/alignment 放置目录 | 视任务而定 | 推理 | `monomer/alignments/` | 由用户预先生成或外部数据库流程生成 |
| `params/finetuning_ptm_2.pt` | 模型权重 | OpenFold finetuning PTM checkpoint | 是 | 推理、微调初始化 | `params/` | SHA256 见 `checksums.sha256` |
| `deepspeed_config.json` | 配置文件 | DeepSpeed 训练或推理相关配置 | 否 | 训练、性能调优 | 仓库根目录 | 可按硬件资源调整 |
| `scripts/` | 辅助脚本目录 | 数据库下载、MSA 预计算、权重转换、缓存生成和工具脚本 | 是 | 推理准备、训练准备 | `scripts/` | 保留 OneScience 示例目录结构 |
| `setup.py` | 安装脚本 | OpenFold 示例包安装入口 | 否 | 环境准备 | 仓库根目录 | 由用户按环境选择是否执行 |
| `thread_sequence.py` | 辅助脚本 | 序列 threading 相关工具 | 否 | 高级用法 | 仓库根目录 | 保留原示例入口 |
| `tools/preflight_check.py` | 预检脚本 | 检查 UTF-8、Manifest、repo_id、关系、命令引用和关键文件 | 是 | 预检 | `tools/` | 上传前和下载后都可运行 |
| `tools/run_minimal_preflight.sh` | 预检包装脚本 | 一键执行模型包轻量预检 | 否 | 预检 | `tools/` | 不运行真实结构预测 |
| `checksums.sha256` | 校验文件 | 记录关键权重和元数据 SHA256 | 是 | 完整性检查 | 仓库根目录 | 标准化时生成 |

## Manifest

本仓库的机器可读 Manifest 位于根目录 `manifest.yaml`，并提供同内容的 `onescience_run_manifest.yaml` 作为 OneScience 网页端运行清单兼容入口。自动运行时必须先解析 Manifest，再根据 `run_matrix.scenarios` 选择预检、用户输入推理或训练入口场景。

如果新增权重、修改命令、调整输入数据约定或移动文件，必须同步更新 `README.md`、`manifest.yaml`、`onescience_run_manifest.yaml` 和 `onescience_relations.yaml`，并重新运行：

```bash
python tools/preflight_openfold.py --repo-root .
```

## 模型 vs 数据集关系

模型仓库 `OneScience/OpenFold` 内置 OpenFold 示例代码、推理入口、训练入口、辅助脚本、示例 FASTA 和 `params/finetuning_ptm_2.pt` 权重。它没有绑定固定的 ModelScope 数据集仓库，因此 `relations.required_datasets` 为空。

实际推理通常需要用户提供 FASTA、预计算 MSA/alignment 目录和模板 mmCIF 目录；如果不使用预计算 alignment，则需要用户在环境中准备 AlphaFold/OpenFold 所需外部数据库，并通过脚本生成 alignment。训练需要用户提供训练 mmCIF、训练 alignment、模板目录、缓存文件和可选验证数据。这些输入属于用户会话数据或外部数据库准备结果，不在本模型仓库中固定声明为必需数据集。

## 文件与下载

下载模型运行包：

```bash
modelscope download --model OneScience/OpenFold --local_dir ./OpenFold_model
cd ./OpenFold_model
```

网页端如果使用 `--cache_dir` 下载，应切换到实际下载后的模型包根目录再执行预检和推理命令。不要只下载 `params/` 权重目录后运行，因为 `run_pretrained_openfold.py`、`train_openfold.py`、`scripts/`、`monomer/inference.sh`、Manifest 和预检脚本也是运行包的一部分。

## 环境安装

在 OneScience 环境中运行 OpenFold 前，建议先安装或加载 bio 域依赖：

```bash
bash install.sh bio
```

推理通常需要 Python、PyTorch、OpenFold/OneScience 生物域模块、模板搜索或预计算 alignment 结果，以及可用 CPU/GPU 资源。预检脚本本身只依赖 Python 标准库和 PyYAML。

## 运行流程

### 1. 环境预检

```bash
python tools/preflight_openfold.py --repo-root .
```

### 2. 下载

```bash
modelscope download --model OneScience/OpenFold --local_dir ./OpenFold_model
cd ./OpenFold_model
```

### 3. 准备输入

最小包预检使用仓库内 `monomer/fasta_dir/6kwc.fasta` 检查目录和权重是否完整。真实推理需要准备以下输入：

```bash
export FASTA_DIR=./monomer/fasta_dir
export OUTPUT_DIR=./monomer
export PRECOMPUTED_ALIGNMENT_DIR=./monomer/alignments
export MMCIF_DIR=/path/to/pdb_mmcif/mmcif_files
```

如果 `PRECOMPUTED_ALIGNMENT_DIR` 为空，需要先用外部数据库和 `scripts/precompute_alignments.py` 或相关脚本生成 alignment。

### 4. 运行前预检

```bash
python tools/preflight_openfold.py --repo-root .
```

### 5. 用户输入推理

```bash
python run_pretrained_openfold.py "$FASTA_DIR" "$MMCIF_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --config_preset model_1_ptm \
  --model_device cuda:0 \
  --data_random_seed 42 \
  --use_precomputed_alignments "$PRECOMPUTED_ALIGNMENT_DIR" \
  --openfold_checkpoint_path params/finetuning_ptm_2.pt
```

也可以阅读并按本地环境修改：

```bash
bash monomer/inference.sh
```

### 6. 训练入口

```bash
python train_openfold.py "$TRAIN_MMCIF_DIR" "$TRAIN_ALIGNMENT_DIR" "$TEMPLATE_MMCIF_DIR" outputs/train "2021-10-10" \
  --deepspeed_config_path deepspeed_config.json \
  --resume_from_ckpt params/finetuning_ptm_2.pt \
  --resume_model_weights_only true
```

训练命令中的训练数据、alignment、模板、缓存和验证数据由用户根据任务准备。执行前应先确认输入目录结构与 `train_openfold.py --help` 中的参数说明一致。

## 输出说明

推理输出由 `--output_dir` 控制。成功运行后，OpenFold 会在输出目录下生成预测结构文件，常见形式包括未松弛结构 PDB 或 ModelCIF 文件，以及在启用 `--save_outputs` 时保存的输出字典。训练输出由 `output_dir` 控制，通常包含 checkpoint、日志和性能记录。

## 预检与诊断

| 问题 | 可能原因 | 处理方式 |
|---|---|---|
| `MODEL_PREFLIGHT_OK` 未出现 | 预检脚本发现缺字段、缺文件或乱码 | 根据错误提示修复后重新运行预检 |
| `repo id mismatch` | README、Manifest、下载命令或关系文件中出现错误 ID | 将模型 ID 统一为 `OneScience/OpenFold` |
| `missing required file` | 下载不完整，或只下载了权重目录 | 重新执行 `modelscope download --model OneScience/OpenFold --local_dir ./OpenFold_model` |
| `ModuleNotFoundError` | 没有加载 OneScience/OpenFold Python 环境 | 安装或加载 OneScience bio 域依赖，并确认 Python 可以导入相关模块 |
| alignment 目录为空 | 用户未准备 MSA 或预计算 alignment | 使用外部数据库和 `scripts/precompute_alignments.py` 生成 alignment，或移除 `--use_precomputed_alignments` 并配置数据库路径 |
| 模板 mmCIF 目录不存在 | `MMCIF_DIR` 指向错误或数据库未准备 | 设置正确的 PDB mmCIF 目录 |
| CUDA 内存不足 | 序列过长、模板或 MSA 过多、设备资源不足 | 改用更大显存设备、减少输入规模，或启用长序列推理相关参数 |

## 限制与适用范围

本资源适合 OpenFold 单体或相应配置预设下的蛋白质结构预测、模型包完整性验证、推理入口复用和训练入口复用。它不内置 AlphaFold/OpenFold 全量数据库，不替用户生成 MSA，也不提供固定训练数据集。真实推理和训练的科学有效性取决于用户提供的 FASTA、alignment、模板、数据库版本和运行配置。

## 引用与许可证

OpenFold 原始方法来自蛋白质结构预测相关论文和开源实现。本仓库按 OneScience 标准运行包整理代码、权重、示例和运行元数据；具体许可证以仓库内原始文件和 ModelScope 页面声明为准。

