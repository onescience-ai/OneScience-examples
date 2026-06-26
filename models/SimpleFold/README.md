# SimpleFold

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 SimpleFold 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/simplefold/" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

SimpleFold 是面向蛋白质结构预测的生成式折叠模型运行包，输入为蛋白质 FASTA 序列，输出为 mmCIF 或 PDB 结构文件。本仓库整理为 OneScience 标准运行包，包含推理脚本、训练入口、Hydra 配置、示例 FASTA、预检脚本、权重链接和机器可读 Manifest。

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

SimpleFold 用通用 Transformer 层和流匹配目标进行蛋白质折叠建模，适合从氨基酸序列预测三维结构，也可作为训练和评测蛋白质折叠模型的 OneScience 示例入口。该标准包保留了原始 OneScience examples 中的 `inference.py`、`train.py`、`train_fsdp.py`、`configs/` 和 `assets/`，并新增 `run_inference.py` 与 `scripts/preflight.py`，方便网页端大模型读取 README 和 Manifest 后执行下载、预检和最小推理。

本次任务的原始数据路径为“无”，因此推理场景不需要下载数据集；训练和评测的数据接口仍保留在配置中，用户如需训练，应按 `configs/data/pdb_sp.yaml` 准备 `./datasets`、`./datasets/tokenized` 和 `./datasets/manifest.json`。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| ModelScope 模型 ID | `OneScience/simplefold/` |
| OneScience 领域 | `bio` |
| 领域标签 | `bio`, `biosciences`, `protein_folding` |
| 任务 | 蛋白质结构预测 |
| 任务标签 | `protein_folding`, `structure_prediction`, `flow_matching` |
| 主平台资源 | `https://modelscope.cn/models/OneScience/simplefold/` |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/biosciences/simplefold` |
| 必需模型文件 | `checkpoints/` 下 SimpleFold 多尺度权重、pLDDT 权重、`ccd.pkl` |
| 必需数据集 | `OneScience/simplefold_dataset`，仅用于训练/评测接口占位 |
| 支持能力 | 预检、推理、训练接口 |
| 最小验证 | `python scripts/preflight.py` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 说明文件 | 人类和大模型的第一入口 | 是 | 全部 | 仓库根目录 | 正文中文 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源、文件、关系、命令和诊断 | 是 | 全部 | 仓库根目录 | 网页端优先读取 |
| `manifest.yaml` | Manifest 文件 | 与 `onescience_run_manifest.yaml` 内容一致，兼容默认文件名 | 是 | 全部 | 仓库根目录 | 自动化工具可解析 |
| `run_inference.py` | 推理入口 | 直接调用本仓库 `inference.py` 执行 FASTA 推理 | 是 | 推理 | 仓库根目录 | 避免依赖 `simplefold` 包名入口 |
| `inference.py` | 原始推理脚本 | 初始化模型、ESM、采样器并保存结构 | 是 | 推理 | 仓库根目录 | 依赖 OneScience bio 环境 |
| `train.py` / `train_fsdp.py` | 训练入口 | 单机或 FSDP 训练入口 | 训练必需 | 训练 | 仓库根目录 | 需要用户准备数据 |
| `configs/` | 配置目录 | Hydra 模型、数据、训练、日志配置 | 是 | 推理、训练、预检 | `configs/` | 未改写原始数据接口 |
| `checkpoints/` | 权重目录 | SimpleFold 多尺度权重、pLDDT 权重、CCD 辅助文件 | 是 | 推理、训练、评测 | `checkpoints/` | 本地整理包中为符号链接；上传时需确保真实文件进入仓库 |
| `MODEL_FILE_MANIFEST.tsv` | 校验清单 | 权重文件名、大小和 SHA256 | 是 | 预检、上传核对 | 仓库根目录 | 来自原始权重目录 |
| `assets/` | 示例结构 | 原始 examples 附带的参考 CIF 和图片 | 否 | 示例、诊断 | `assets/` | 不作为训练数据 |
| `examples/minimal.fasta` | 示例输入 | 最小 FASTA 推理输入 | 是 | 推理 | `examples/minimal.fasta` | 可替换为用户 FASTA |
| `scripts/preflight.py` | 预检脚本 | 检查 Manifest、配置、权重、样例和训练数据接口 | 是 | 预检 | `scripts/preflight.py` | 不执行重推理 |

## Manifest

机器可读 Manifest 位于仓库根目录：

- `onescience_run_manifest.yaml`
- `manifest.yaml`

两个文件内容一致。大模型应先读取 README 的“文件说明”和本章节，再解析 Manifest。修改模型文件、命令、关系或配置后，必须同步更新这两个 Manifest 和 `onescience_relations.yaml`。

## 模型 vs 数据集关系

本模型仓库的 ModelScope ID 必须保持为 `OneScience/simplefold/`。关联数据集 ID 必须保持为 `OneScience/simplefold_dataset`。

推理场景不需要数据集，只需要模型权重、FASTA 输入、CCD 文件和 OneScience bio 环境。训练/评测场景通过 `relations.required_datasets` 显式声明 `OneScience/simplefold_dataset`，但本次原始数据路径为“无”，数据集仓库仅提供 schema 和空数据接口；用户训练时应自行补齐兼容数据。

## 文件与下载

下载模型：

```bash
modelscope download --model OneScience/simplefold/
```

下载关联数据集元信息：

```bash
modelscope download --dataset OneScience/simplefold_dataset
```

如果网页端或脚本使用 `--cache_dir`，下载完成后运行 `python scripts/preflight.py` 前，必须把 `cwd` 切换到实际下载后的模型包根目录，也就是包含 `README.md`、`onescience_run_manifest.yaml` 和 `scripts/preflight.py` 的目录。

## 环境安装

网站环境已部署 OneScience 时，优先直接运行预检。若缺少 bio 领域依赖，在 OneScience 根目录执行：

```bash
bash install.sh bio
```

推理需要 PyTorch 或 MLX 后端。ESM-2 3B 权重建议放在：

```bash
$ONESCIENCE_MODELS_DIR/esm_models/esm2_t36_3B_UR50D.pt
```

也可以通过 `SIMPLEFOLD_ESM2_MODEL_PATH` 指定本地权重。

## 运行流程

预检：

```bash
python scripts/preflight.py
```

最小推理：

```bash
python run_inference.py --simplefold_model simplefold_100M --ckpt_dir checkpoints --fasta_path examples/minimal.fasta --output_dir outputs/minimal_inference --num_steps 10 --tau 0.01 --nsample_per_protein 1 --backend torch
```

训练接口：

```bash
python train.py
```

训练前必须准备：

- `./datasets`
- `./datasets/tokenized`
- `./datasets/manifest.json`

## 预检与诊断

| 现象 | 常见原因 | 处理方式 |
|---|---|---|
| `ModuleNotFoundError` | OneScience bio 环境或依赖未安装 | 执行 `bash install.sh bio`，或切换到可 import `onescience` 的环境 |
| `No such file or directory` | 权重、FASTA、CCD 或训练数据缺失 | 运行 `python scripts/preflight.py`，按 Manifest 的 `files` 补齐 |
| `Local ESM-2 3B weights not found` | ESM-2 权重不在默认路径 | 设置 `SIMPLEFOLD_ESM2_MODEL_PATH` 或放入 `$ONESCIENCE_MODELS_DIR/esm_models/` |
| `CUDA out of memory` | 模型规模或序列过大 | 先用 `simplefold_100M`，减少 `num_steps` 或换更大显存设备 |

## 输出说明

最小推理输出目录为：

```text
outputs/minimal_inference/predictions_simplefold_100M/
```

训练输出通常位于 Lightning/Hydra 日志目录，例如 `logs/train/` 或运行时生成的 `outputs/` 子目录。

## 限制与适用范围

本仓库适用于 OneScience bio 环境中的 SimpleFold 推理和训练接口验证。本次不包含真实训练数据；`checkpoints/` 在当前整理目录中使用符号链接指向共享权重，正式上传 ModelScope 前应确认上传工具会包含真实权重文件，或使用跟随符号链接的上传方式。

## 引用与许可证

SimpleFold 原始实现声明为 MIT License。引用论文和上游实现时，请参考原始 SimpleFold 项目说明。
