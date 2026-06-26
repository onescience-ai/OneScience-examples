# ESMFold

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 ESMFold 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/esmfold/" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

ESMFold 是面向蛋白质单序列结构预测的模型运行包。该标准仓库把 OneScience 中 `examples/biosciences/esm` 的 ESMFold 推理脚本、最小 FASTA 样例、权重入口和运行说明整理为 ModelScope 可下载包，可从 FASTA 输入生成 PDB 结构文件。

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

本资源是 `OneScience/esmfold/` 模型标准运行包，领域为 `bio`。它接收蛋白质 FASTA 序列，调用 OneScience 的 ESM/ESMFold 实现和 `esmfold_3B_v1.pt` 权重，输出每条序列对应的 PDB 文件，并在日志中给出 pLDDT、pTM 和推理进度。

当前包支持最小预检和最小推理，不声明训练、微调或评测能力。原始数据目录 `/public/share/sugonhpcapp01/onestore/onedatasets/esmfold/data` 为空，因此本次整理将可用资源明确为 `weight/` 权重集合，并通过 `OneScience/esmfold_dataset` 与模型包双向关联。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | bio |
| 领域标签 | bio, protein, structure_prediction |
| 任务 | protein_structure_prediction |
| 任务标签 | esmfold, protein_folding, fasta_to_pdb |
| 主平台资源 | https://modelscope.cn/models/OneScience/esmfold/ |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/biosciences/esm` |
| 必需模型文件 | `checkpoints/esmfold_3B_v1.pt` |
| 必需数据集 | `OneScience/esmfold_dataset` |
| 支持能力 | 预检、推理 |
| 最小验证 | `python scripts/preflight.py --dataset-root ../../dataset/bio_esmfold_dataset` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 说明文档 | 人类和大模型运行入口 | 是 | 全部能力 | 模型包根目录 | 中文为主 |
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明 | 是 | 全部能力 | `manifest.yaml` | 默认 Manifest 路径 |
| `onescience_run_manifest.yaml` | Manifest 兼容副本 | 供网页端按固定名称读取 | 是 | 全部能力 | `onescience_run_manifest.yaml` | 内容与 `manifest.yaml` 一致 |
| `conf/config.yaml` | 配置文件 | 声明 FASTA、输出目录、权重目录和数据集位置 | 是 | 预检、推理 | `conf/config.yaml` | 已适配标准包相对路径 |
| `scripts/preflight.py` | 预检脚本 | 检查 YAML、FASTA、权重大小、SHA256 和关联数据集 | 是 | 预检 | `scripts/preflight.py` | 不加载 3B 模型 |
| `scripts/fold.py` | 推理脚本 | FASTA 到 PDB 的 ESMFold 推理入口 | 是 | 推理 | `scripts/fold.py` | 依赖 OneScience bio 环境 |
| `scripts/extract.py` | 辅助脚本 | ESM-2 表征提取入口 | 否 | 可选表征提取 | `scripts/extract.py` | 非默认场景 |
| `data/sample/few_proteins.fasta` | 样例数据 | 最小推理 FASTA 输入 | 是 | 预检、推理 | `data/sample/few_proteins.fasta` | SHA256 已记录 |
| `checkpoints/esmfold_3B_v1.pt` | 模型权重 | ESMFold v1 推理 checkpoint | 是 | 预检、推理 | `checkpoints/esmfold_3B_v1.pt` | SHA256 已记录 |
| `metadata/ONESCIENCE_ESM_EXAMPLES_README.md` | 来源说明 | OneScience ESM examples README 备份 | 否 | 溯源 | `metadata/` | 说明原始运行方式 |

## Manifest

默认机器可读文件是仓库根目录的 `manifest.yaml`，兼容文件是 `onescience_run_manifest.yaml`。修改运行脚本、文件路径、下载命令、资源 ID 或模型-数据集关系时，必须同步更新这两个文件，并重新执行 YAML 解析和 command_refs 校验。

## 模型 vs 数据集关系

模型资源 ID 必须保持为 `OneScience/esmfold/`。数据集资源 ID 必须保持为 `OneScience/esmfold_dataset`。模型 Manifest 的 `relations.required_datasets` 指向 `OneScience/esmfold_dataset`，数据集 Manifest 的 `relations.compatible_models` 反向指向 `OneScience/esmfold/`，两端都包含完整 `resource_ref`。

推理场景 `esmfold_sample_inference` 需要模型包中的 `scripts/fold.py`、`data/sample/few_proteins.fasta`、`checkpoints/esmfold_3B_v1.pt`，并要求数据集包中存在 `weight/esmfold_3B_v1.pt` 以完成一致性校验。

## 文件与下载

下载模型包：

```bash
modelscope download --model OneScience/esmfold/ --local_dir ./bio_esmfold
```

下载数据集包：

```bash
modelscope download --dataset OneScience/esmfold_dataset --local_dir ./bio_esmfold_dataset
```

如果网页端使用 `--cache_dir` 下载，运行前必须切换到实际下载后的模型包根目录，也就是包含 `README.md`、`manifest.yaml`、`conf/`、`scripts/` 的目录。

## 环境安装

需要 OneScience bio 环境，并确保 Python 可以导入 `onescience.models.esm` 和 `onescience.datapipes.esm`。在完整 OneScience 源码环境中通常先加载环境变量，再进入模型包根目录运行命令。

```bash
# 示例：在 OneScience 环境已安装后执行
python -c "import onescience.models.esm, onescience.datapipes.esm"
```

## 运行流程

建议目录放置如下：

```text
session_workdir/
├── model/bio_esmfold/
└── dataset/bio_esmfold_dataset/
```

预检：

```bash
cd session_workdir/model/bio_esmfold
python scripts/preflight.py --dataset-root ../../dataset/bio_esmfold_dataset
```

CPU 最小推理：

```bash
cd session_workdir/model/bio_esmfold
python scripts/fold.py \
  -i data/sample/few_proteins.fasta \
  -o outputs/esmfold_pdb \
  --model-dir . \
  --chunk-size 128 \
  --cpu-only
```

有 GPU 时可以去掉 `--cpu-only`，显存不足时降低 `--max-tokens-per-batch` 或把 `--chunk-size` 调整为 `64`、`32`。

## 预检与诊断

预检脚本会检查：

| 检查项 | 说明 |
|---|---|
| YAML 可解析 | 读取 `manifest.yaml` 和 `conf/config.yaml` |
| 资源 ID | 模型 ID 等于 `OneScience/esmfold/`，数据集 ID 等于 `OneScience/esmfold_dataset` |
| FASTA | `data/sample/few_proteins.fasta` 可读且包含记录 |
| 权重 | `checkpoints/esmfold_3B_v1.pt` 存在、大小和 SHA256 匹配 |
| 数据集 | `weight/esmfold_3B_v1.pt` 存在且大小与模型 checkpoint 一致 |

常见错误：

| 现象 | 诊断 | 处理 |
|---|---|---|
| `Dataset weight not found` | 数据集未下载或路径不对 | 下载 `OneScience/esmfold_dataset`，或用 `--dataset-root` 指定 |
| `ModuleNotFoundError: onescience` | 未加载 OneScience 环境 | 安装或加载 OneScience bio 环境 |
| `CUDA out of memory` | 显存不足 | 降低 batch token、减小 chunk size 或使用 CPU |
| 找不到 `checkpoints/esmfold_3B_v1.pt` | 当前 cwd 不是模型包根目录或下载不完整 | 切换到实际模型包根目录并重新下载 |

## 输出说明

推理输出写入 `outputs/esmfold_pdb/`。每条 FASTA 记录生成一个 `{header}.pdb` 文件，日志包含序列长度、pLDDT、pTM、耗时和完成数量。

## 限制与适用范围

本包只提供 ESMFold 最小推理和预检。原始数据路径中没有训练/评测样本切分，因此不声明训练、微调和评测能力。CPU 推理可用于连通性验证，但 3B 模型在 CPU 上可能很慢。

## 引用与许可证

ESM/ESMFold 原始代码使用 MIT License。本标准包保留来源说明并面向 OneScience ModelScope 自动运行场景整理。
