# AlphaGenome

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，进入 OneScience/AlphaGenome 模型资源</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/AlphaGenome" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="OneScience">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

AlphaGenome 是面向 DNA 序列调控功能预测的基因组模型。它以人类或小鼠参考基因组区间、DNA 序列或变异位点为输入，输出染色质可及性、转录起始、RNA 表达、转录因子结合、组蛋白修饰、三维基因组接触图谱和剪接相关信号等多模态预测结果。

本仓库是 OneScience 标准运行包，不是孤立权重仓库。仓库同时包含 AlphaGenome 示例入口脚本、本地 Orbax/OCDBT checkpoint、运行说明、Manifest 和预检脚本；下载后可以在 OneScience 环境中直接以仓库根目录作为工作目录运行预检、推理、变异评分或轨迹预测评估。参考基因组 FASTA、VCF 变异文件和评测 TFRecord 数据不随模型仓库固定绑定，按用户任务自行提供。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| ModelScope ID | `OneScience/AlphaGenome` |
| OneScience 领域 | `bio` |
| 领域标签 | `genomics`, `dna`, `regulatory_genomics` |
| 任务 | DNA 调控功能预测、变异效应评分、轨迹预测评估 |
| 主平台资源 | https://modelscope.cn/models/OneScience/AlphaGenome |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/biosciences/alphagenome` |
| 支持能力 | 预检、推理、变异评分、轨迹预测评估、微调入口 |
| 必需模型文件 | `checkpoints/alphagenome-all-folds/` |
| 必需数据集 | 无固定 ModelScope 数据集依赖；输入由用户提供 |
| 最小验证 | `python preflight_alphagenome.py --model_dir checkpoints/alphagenome-all-folds` |

能力和命令要求：

| 能力 | 命令 |
|---|---|
| `preflight` | `python preflight_alphagenome.py --model_dir checkpoints/alphagenome-all-folds` |
| `inference` | `python run_inference.py --model_dir checkpoints/alphagenome-all-folds --fasta_path /path/to/GRCh38.fa --chromosome chr19 --start 10587331 --end 11635907 --output_dir outputs/inference` |
| `variant_scoring` | `python run_variant_scoring.py --model_dir checkpoints/alphagenome-all-folds --fasta_path /path/to/GRCh38.fa --output_dir outputs/variant` |
| `evaluate` | `python run_track_prediction_eval.py --model_dir checkpoints/alphagenome-all-folds --data_dir /path/to/alphagenome_tfrecords --output_path outputs/track/eval_results.csv` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 文档 | 面向用户和大模型的运行说明 | 是 | 全部能力 | 仓库根目录 | 本文件 |
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、关系、命令和输出 | 是 | 全部能力 | 仓库根目录 | 修改运行包后必须同步更新 |
| `preflight_alphagenome.py` | 预检脚本 | 检查入口脚本、checkpoint 结构、可选依赖和用户输入文件 | 是 | 预检 | 仓库根目录 | 不加载完整模型 |
| `validate_modelscope_package.py` | 标准包校验脚本 | 检查 YAML、README 章节、repo_id、command_ref、relations 和编码 | 是 | 上传前校验 | 仓库根目录 | 上传前和下载后均可运行 |
| `run_inference.py` | 推理脚本 | 对指定基因组区间运行 AlphaGenome 预测 | 是 | 推理 | 仓库根目录 | 支持本地 checkpoint |
| `run_variant_scoring.py` | 变异评分脚本 | 对内置或 VCF 中的 SNV 进行功能影响评分 | 是 | 变异评分 | 仓库根目录 | 使用本地 checkpoint 时需要 FASTA |
| `run_track_prediction_eval.py` | 评测脚本 | 在 AlphaGenome TFRecord 评测数据上计算轨迹预测指标 | 是 | 评测 | 仓库根目录 | 评测数据由用户提供 |
| `run_finetuning.py` | 微调脚本 | 提供 OneScience AlphaGenome 微调入口 | 可选 | 微调 | 仓库根目录 | 需要用户自备训练数据 |
| `inference.sh` | Shell 示例 | 推理命令示例 | 可选 | 推理 | 仓库根目录 | 可按路径修改 |
| `run_variant.sh` | Shell 示例 | 变异评分命令示例 | 可选 | 变异评分 | 仓库根目录 | 可按路径修改 |
| `run_track.sh` | Shell 示例 | 轨迹评估命令示例 | 可选 | 评测 | 仓库根目录 | 可按路径修改 |
| `checkpoints/alphagenome-all-folds/` | 模型权重 | AlphaGenome all-folds Orbax/OCDBT checkpoint | 是 | 推理、变异评分、评测、微调初始化 | 仓库根目录下同名路径 | 约 701M，含 `_CHECKPOINT_METADATA`、`_METADATA`、`manifest.ocdbt` 和 `ocdbt.process_0/` |

## Manifest

Manifest 文件位于仓库根目录 `manifest.yaml`。它是大模型执行下载、预检、推理和诊断的主依据，包含 `resource`、`platform_resource.primary`、`runtime_package`、`files`、`relations`、`run_matrix`、`commands`、`expected_outputs`、`diagnostics` 和 `domain_extension` 等字段。

如果修改仓库 ID、权重目录、入口脚本、数据放置方式或运行命令，必须同步更新 `manifest.yaml`，并运行：

```bash
python validate_modelscope_package.py --root .
python preflight_alphagenome.py --model_dir checkpoints/alphagenome-all-folds
```

## 模型 vs 数据集关系

本模型仓库没有固定绑定的 ModelScope 数据集仓库，`relations.required_datasets` 为空。推理和变异评分通常需要用户提供参考基因组 FASTA 及 `.fai` 索引；变异评分可以额外提供 VCF；轨迹预测评估可以通过 `--data_dir` 指定 AlphaGenome TFRecord 评测数据目录。由于这些输入依任务和物种而变化，Manifest 中将它们声明为用户提供的运行输入，而不是固定数据集依赖。

## 文件与下载

使用 ModelScope CLI 下载完整标准运行包：

```bash
modelscope download --model OneScience/AlphaGenome --local_dir ./AlphaGenome
cd ./AlphaGenome
```

如果使用 `--cache_dir` 下载，请切换到实际下载后的模型包根目录，再执行预检和运行命令。模型权重已经放在 `checkpoints/alphagenome-all-folds/`，无需再从 Kaggle Hub 下载。

## 环境安装

本仓库依赖 OneScience 生物信息环境和 AlphaGenome 相关 Python 依赖。推荐在已经安装 OneScience 的环境中运行：

```bash
bash install.sh bio
```

若当前环境没有 OneScience 源码，请先参考上方 OneScience 官方文档完成安装。预检脚本会提示 `onescience`、`jax`、`orbax.checkpoint`、`absl`、`numpy`、`pandas`、`tensorflow` 等可选依赖是否可导入。

## 运行流程

### 1. 环境预检

```bash
python preflight_alphagenome.py --model_dir checkpoints/alphagenome-all-folds
```

### 2. 准备输入

将参考基因组 FASTA 和 `.fai` 索引放到用户可访问路径，例如：

```text
/path/to/GRCh38.fa
/path/to/GRCh38.fa.fai
```

如运行变异评分，可准备 VCF 文件；如运行评测，可准备 AlphaGenome TFRecord 数据目录。

### 3. 运行推理

```bash
python run_inference.py \
  --model_dir checkpoints/alphagenome-all-folds \
  --fasta_path /path/to/GRCh38.fa \
  --chromosome chr19 \
  --start 10587331 \
  --end 11635907 \
  --output_dir outputs/inference
```

### 4. 运行变异评分

```bash
python run_variant_scoring.py \
  --model_dir checkpoints/alphagenome-all-folds \
  --fasta_path /path/to/GRCh38.fa \
  --output_dir outputs/variant
```

如需使用自定义变异：

```bash
python run_variant_scoring.py \
  --model_dir checkpoints/alphagenome-all-folds \
  --fasta_path /path/to/GRCh38.fa \
  --vcf_path /path/to/variants.vcf \
  --output_dir outputs/variant
```

### 5. 运行轨迹预测评估

```bash
python run_track_prediction_eval.py \
  --model_dir checkpoints/alphagenome-all-folds \
  --data_dir /path/to/alphagenome_tfrecords \
  --output_path outputs/track/eval_results.csv
```

## 输出说明

推理输出默认写入 `outputs/inference/`，每种预测类型保存为独立 `.npy` 文件，例如 `atac.npy`、`dnase.npy`、`cage.npy`、`rna_seq.npy`、`chip_tf.npy` 和 `chip_histone.npy`。变异评分输出默认写入 `outputs/variant/`，包含每个变异的评分 CSV 和 `variant_scoring_summary.csv`。轨迹预测评估输出为 CSV 文件，包含数据束、指标名称和指标值。

## 预检与诊断

常见问题和处理方式：

| 问题 | 可能原因 | 处理方式 |
|---|---|---|
| 找不到 `_CHECKPOINT_METADATA` | 模型包未完整下载或工作目录不对 | 确认当前目录是模型包根目录，并检查 `checkpoints/alphagenome-all-folds/` |
| 找不到 `manifest.ocdbt` 或 `ocdbt.process_0/` | OCDBT checkpoint 不完整 | 重新下载 `OneScience/AlphaGenome`，不要只复制部分权重 |
| 使用 `--model_dir` 时提示需要 FASTA | 本地 checkpoint 未绑定参考基因组 | 提供 `--fasta_path`，并确保 `.fai` 索引存在 |
| `ModuleNotFoundError: onescience` | 未进入 OneScience Python 环境 | 安装或激活 OneScience bio 环境后重试 |
| `jax` 或 `tensorflow` 导入失败 | AlphaGenome 依赖未安装 | 按 OneScience bio 环境安装依赖 |
| 评测找不到 TFRecord | 未提供评测数据目录 | 使用 `--data_dir /path/to/alphagenome_tfrecords` 指向用户数据 |

## 限制与适用范围

本仓库提供 AlphaGenome 本地 checkpoint 与 OneScience 示例入口，不随仓库提供参考基因组、VCF、完整训练集或完整评测集。模型适用于人类和小鼠基因组调控功能相关任务；输入区间长度、物种、参考基因组版本和评测数据格式应与脚本参数和 OneScience AlphaGenome 实现保持一致。

## 引用与许可证

AlphaGenome 示例脚本保留原始 Apache License 2.0 许可头。使用本模型时请同时遵守 AlphaGenome/OneScience 代码、模型权重来源和相关数据的许可证要求。
