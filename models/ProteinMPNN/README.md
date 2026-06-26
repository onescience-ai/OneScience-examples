---
frameworks:
- ""
tasks: []
---
# ProteinMPNN

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 ProteinMPNN 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/ProteinMPNN" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

ProteinMPNN 是面向蛋白质骨架到氨基酸序列设计的深度学习模型。给定 PDB 蛋白质结构、链选择或由辅助脚本解析得到的 JSONL 输入，模型可以生成与目标骨架匹配的候选序列，也可以对已有骨架和序列组合进行打分、输出条件概率或无条件概率。

本仓库是 OneScience 标准运行包形式的 ProteinMPNN 模型资源，上传内容不是孤立权重，而是来自 `examples/biosciences/ProteinMPNN` 的代码、示例输入、辅助脚本、训练脚本、权重目录和预检脚本。网页端大模型下载本仓库后，应以仓库根目录作为运行工作目录，并优先读取 `manifest.yaml` 生成下载、预检和运行计划。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| ModelScope ID | `OneScience/ProteinMPNN` |
| OneScience 领域 | `bio` |
| 领域标签 | `bio`, `protein`, `protein_design` |
| 任务 | 蛋白质骨架到序列设计 |
| 任务标签 | `protein_sequence_design`, `backbone_conditioned_generation`, `protein_scoring` |
| 主平台资源 | https://modelscope.cn/models/OneScience/ProteinMPNN |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/biosciences/ProteinMPNN` |
| 支持能力 | 预检、推理、打分、训练入口、评测入口 |
| 必需模型文件 | `vanilla_model_weights/v_48_020.pt`；其他权重用于不同噪声或 CA-only/soluble 场景 |
| 必需数据集 | 训练和评测关联 `OneScience/proteinmpnn`；最小推理使用仓库内示例 PDB，不需要外部数据集 |
| 最小验证 | `python tools/preflight_proteinmpnn.py --repo-root .` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 说明文件 | 面向人类和大模型的运行入口 | 是 | 全部能力 | 仓库根目录 | 读取后继续解析 `manifest.yaml` |
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、关系、命令和输出 | 是 | 全部能力 | 仓库根目录 | 修改运行包后必须同步更新 |
| `onescience_run_manifest.yaml` | Manifest 副本 | 兼容 OneScience 网页端对运行清单文件名的读取 | 是 | 全部能力 | 仓库根目录 | 内容与 `manifest.yaml` 保持一致 |
| `onescience_relations.yaml` | 关系文件 | 显式声明模型与数据集 `OneScience/proteinmpnn` 的关系 | 是 | 解析关系 | 仓库根目录 | 与 Manifest 的 `relations` 保持一致 |
| `protein_mpnn_run.py` | 推理脚本 | 执行蛋白质序列设计、打分和概率输出 | 是 | 推理、打分 | 仓库根目录 | 依赖 OneScience 中的 `onescience.models.proteinmpnn` |
| `helper_scripts/` | 辅助脚本目录 | 解析 PDB、固定链、固定残基、PSSM、偏置和绑定位点 | 是 | 推理准备 | 仓库根目录 | 保留 OneScience 示例目录结构 |
| `infer_examples/` | 示例命令目录 | 提供不同推理场景的 shell 示例 | 否 | 推理参考 | 仓库根目录 | 标准运行建议使用 Manifest 中命令 |
| `train/` | 训练脚本目录 | 提供训练入口和训练辅助工具 | 训练时必需 | 训练 | 仓库根目录 | 训练需要下载 `OneScience/proteinmpnn` 数据集 |
| `inputs/PDB_monomers/pdbs/5L33.pdb` | 示例输入 | 最小推理验证使用的单体 PDB | 是 | 最小推理 | 仓库根目录 | 不依赖外部数据集 |
| `inputs/PDB_complexes/` | 示例输入目录 | 多链复合物示例 PDB | 否 | 推理示例 | 仓库根目录 | 用于复杂场景测试 |
| `inputs/PSSM_inputs/` | 示例 PSSM 输入 | PSSM 偏置示例文件 | 否 | PSSM 推理 | 仓库根目录 | 与 `infer_examples/submit_example_pssm.sh` 对应 |
| `vanilla_model_weights/` | 模型权重目录 | 标准 ProteinMPNN 权重 | 是 | 推理、打分、评测 | 仓库根目录 | 包含 `v_48_002.pt`、`v_48_010.pt`、`v_48_020.pt`、`v_48_030.pt` |
| `soluble_model_weights/` | 模型权重目录 | 可溶蛋白版本权重和排除列表 | 可选 | 推理、打分 | 仓库根目录 | 使用 `--use_soluble_model` |
| `ca_model_weights/` | 模型权重目录 | CA-only ProteinMPNN 权重 | 可选 | CA-only 推理 | 仓库根目录 | 使用 `--ca_only` |
| `tools/preflight_check.py` | 预检脚本 | 检查 UTF-8、Manifest、repo_id、关系、命令引用和关键文件 | 是 | 预检 | `tools/` | 上传前和下载后都可运行 |
| `tools/run_minimal_inference.sh` | 运行脚本 | 使用内置示例 PDB 和标准权重执行最小推理 | 否 | 推理 | `tools/` | 需要已安装 OneScience Python 环境 |
| `checksums.sha256` | 校验文件 | 记录关键权重和元数据 SHA256 | 是 | 完整性检查 | 仓库根目录 | 由标准化目录生成 |

## Manifest

本仓库的机器可读 Manifest 位于根目录 `manifest.yaml`，并提供同内容的 `onescience_run_manifest.yaml` 作为 OneScience 网页端运行清单兼容入口。自动运行时必须先解析 Manifest，再根据 `run_matrix.scenarios` 选择最小推理、训练或评测场景。

如果新增权重、修改命令、调整数据集依赖或移动文件，必须同步更新 `README.md`、`manifest.yaml`、`onescience_run_manifest.yaml` 和 `onescience_relations.yaml`，并重新运行：

```bash
python tools/preflight_proteinmpnn.py --repo-root .
```

## 模型 vs 数据集关系

模型仓库 `OneScience/ProteinMPNN` 内置了最小推理所需的示例 PDB 和全部三类权重，因此最小推理不需要下载外部数据集。训练、评测和复现完整数据流程需要关联数据集仓库 `OneScience/proteinmpnn`，数据集下载后建议放在会话工作区的 `./proteinmpnn_dataset`。

Manifest 中的 `relations.required_datasets` 使用完整 `resource_ref` 指向 `OneScience/proteinmpnn`，数据集仓库应通过 `relations.compatible_models` 反向声明适配 `OneScience/ProteinMPNN`。

## 文件与下载

下载模型运行包：

```bash
modelscope download --model OneScience/ProteinMPNN --local_dir ./ProteinMPNN_model
cd ./ProteinMPNN_model
```

如需训练或评测，另行下载关联数据集：

```bash
modelscope download --dataset OneScience/proteinmpnn --local_dir ./proteinmpnn_dataset
```

网页端如果使用 `--cache_dir` 下载，应切换到实际下载后的模型包根目录再执行预检和推理命令。不要只下载权重目录后运行，因为辅助脚本、示例输入、Manifest 和预检脚本也是运行包的一部分。

## 环境安装

在 OneScience 环境中运行 ProteinMPNN 前，建议先安装或加载 bio 域依赖：

```bash
bash install.sh bio
```

运行推理需要 Python、PyTorch、NumPy 和 OneScience 源码中的 `onescience.models.proteinmpnn` 模块。预检脚本本身只依赖 Python 标准库和 PyYAML。

## 运行流程

### 1. 环境预检

```bash
python tools/preflight_proteinmpnn.py --repo-root .
```

### 2. 下载

```bash
modelscope download --model OneScience/ProteinMPNN --local_dir ./ProteinMPNN_model
cd ./ProteinMPNN_model
```

### 3. 准备数据

最小推理直接使用仓库内 `inputs/PDB_monomers/pdbs/5L33.pdb`。训练或评测需要下载 `OneScience/proteinmpnn`，并把数据集根目录通过环境变量传入：

```bash
export PROTEINMPNN_DATA_ROOT=../proteinmpnn_dataset
```

### 4. 运行前预检

```bash
python tools/preflight_proteinmpnn.py --repo-root .
```

### 5. 最小推理

```bash
bash tools/run_minimal_inference.sh
```

等价命令如下：

```bash
python protein_mpnn_run.py \
  --pdb_path inputs/PDB_monomers/pdbs/5L33.pdb \
  --pdb_path_chains A \
  --path_to_model_weights vanilla_model_weights \
  --model_name v_48_020 \
  --out_folder outputs/modelscope_minimal \
  --num_seq_per_target 2 \
  --sampling_temp "0.1" \
  --seed 37 \
  --batch_size 1
```

### 6. 训练入口

```bash
python train/training.py --path_for_training_data "$PROTEINMPNN_DATA_ROOT"
```

训练入口依赖 `OneScience/proteinmpnn` 的数据结构。执行前应先阅读该数据集仓库的 README 和 Manifest，并确认数据根目录包含训练脚本所需的列表、PDB 或解析后文件。

## 输出说明

最小推理默认输出到 `outputs/modelscope_minimal`。成功运行后通常会出现 `seqs/` 目录和 FASTA 结果文件，文件头中包含目标结构名、score、global_score、model_name 和 seed 等信息。打分或概率模式会额外输出 `.npy` 或 `.npz` 文件，具体取决于命令参数。

## 预检与诊断

常见问题和处理方式：

| 问题 | 可能原因 | 处理方式 |
|---|---|---|
| `missing required file` | 下载不完整，或只下载了权重目录 | 重新执行 `modelscope download --model OneScience/ProteinMPNN --local_dir ./ProteinMPNN_model` |
| `repo id mismatch` | README、Manifest 或关系文件中出现错误 repo_id | 将模型 ID 统一为 `OneScience/ProteinMPNN`，数据集 ID 统一为 `OneScience/proteinmpnn` |
| `MODEL_PREFLIGHT_OK` 未出现 | 预检脚本发现缺字段、缺文件或乱码 | 根据错误提示修复后重新运行预检 |
| `ModuleNotFoundError: onescience` | 没有加载 OneScience Python 环境 | 安装或加载 OneScience，并确认 Python 可以导入 `onescience` |
| `No such file or directory: *.pt` | 权重目录路径传错 | 使用 `--path_to_model_weights vanilla_model_weights` 或对应绝对路径 |
| CUDA 内存不足 | 批量大小或序列长度过大 | 降低 `--batch_size` 或缩短输入结构 |

## 限制与适用范围

本资源适合蛋白质骨架条件序列设计、候选序列打分和概率输出。模型不直接完成结构预测、分子动力学模拟或湿实验验证。训练和完整评测需要配套数据集 `OneScience/proteinmpnn`，最小推理只验证模型包结构、权重可定位和示例命令可生成序列。

## 引用与许可证

ProteinMPNN 原始方法来自蛋白质序列设计相关论文和开源实现。本仓库按 OneScience 标准运行包整理代码、权重、示例和运行元数据；具体许可证以仓库内原始文件和 ModelScope 页面声明为准。

