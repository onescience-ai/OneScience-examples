# Protenix

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 Protenix 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/protenix/" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

Protenix 是面向蛋白质、核酸、配体等生物分子复合物结构预测的 AlphaFold3-like 模型。本仓库是 `OneScience/protenix/` 的 OneScience ModelScope 标准模型包，包含 Protenix 运行代码、推理/训练/微调脚本、样例输入、预检脚本和 `model_v0.5.0.pt` 权重。

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

Protenix 用于预测生物分子复合物三维结构，输入通常是 JSON 描述的序列、配体和约束信息，并可结合 A3M MSA、CCD 组件字典和 RDKit 缓存完成特征化；输出为预测结构 CIF 文件和置信度 JSON。

本标准包已从 OneScience `examples/biosciences/protenix` 整理为可下载运行目录。模型默认读取相邻的 `OneScience/protenix_dataset` 数据集目录，最小验证场景使用 `infer_datasets/7r6r.json` 和仓库内样例 MSA。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| ModelScope 模型 ID | `OneScience/protenix/` |
| OneScience 领域 | `bio` |
| 领域标签 | `bio`, `protein_structure`, `molecular_structure` |
| 任务 | 生物分子复合物结构预测 |
| 任务标签 | `structure_prediction`, `protein_folding`, `molecular_complex` |
| 主平台资源 | https://modelscope.cn/models/OneScience/protenix/ |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/biosciences/protenix` |
| 必需模型文件 | `checkpoints/model_v0.5.0.pt` |
| 必需数据集 | `OneScience/protenix_dataset` |
| 支持能力 | 推理、训练、微调 |
| 最小验证 | `python scripts/preflight_protenix.py --model-root . --data-root ../bio_protenix_dataset` |

| 能力 | 必须提供 |
|---|---|
| `inference` | `bash inference_unified_demo.sh`，输入 `infer_datasets/7r6r.json`，输出 `output_unified/7r6r/seed_101/predictions/` |
| `train` | `bash train_demo.sh`，读取 `OneScience/protenix_dataset` 完整训练数据，输出 `output/` |
| `finetune` | `bash finetune_demo.sh`，加载 `checkpoints/model_v0.5.0.pt` 和 `ft_datasets/finetune_subset.txt`，输出 `output/` |
| `evaluate` | 当前未提供独立标准评测入口 |
| `visualize` | 当前未提供标准可视化入口，可用 PyMOL 打开输出 CIF |
| `deploy` | 当前未提供部署入口 |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明模型文件、数据集关系、下载方式和运行命令 | 是 | 全部能力 | 模型仓库根目录 | 修改运行包时必须同步更新 |
| `onescience_run_manifest.yaml` | Manifest 副本 | 供网页端按固定文件名读取 | 是 | 全部能力 | 模型仓库根目录 | 内容与 `manifest.yaml` 保持一致 |
| `checkpoints/model_v0.5.0.pt` | 权重 | Protenix v0.5.0 预训练权重 | 是 | 推理、微调 | `checkpoints/model_v0.5.0.pt` | SHA256 写在 Manifest 中 |
| `configs/inference_config.yaml` | 配置 | 推理输入、输出目录、权重路径和数据依赖 | 是 | 推理、预检 | `configs/inference_config.yaml` | 已适配本标准包相对路径 |
| `inference_unified_demo.sh` | 脚本 | 最小统一接口推理入口 | 是 | 推理 | 模型仓库根目录 | 默认读取 `../bio_protenix_dataset` |
| `train_demo.sh` | 脚本 | 单卡训练入口 | 是 | 训练 | 模型仓库根目录 | 需要完整数据集 |
| `finetune_demo.sh` | 脚本 | 单卡微调入口 | 是 | 微调 | 模型仓库根目录 | 默认加载仓库内权重 |
| `scripts/preflight_protenix.py` | 脚本 | 检查配置、权重、样例输入和数据集路径 | 是 | 预检 | `scripts/preflight_protenix.py` | 可选 `--full-checksum` |
| `infer_datasets/` | 样例数据 | 7r6r、7pzb、7wux 等推理样例和 MSA | 是 | 推理 | `infer_datasets/` | 最小验证使用 7r6r |
| `runner/` | 运行代码 | Protenix 推理、训练和工具代码 | 是 | 推理、训练、微调 | `runner/` | 来自 OneScience 示例目录 |

## Manifest

标准 Manifest 位于仓库根目录 `manifest.yaml`。本仓库同时提供 `onescience_run_manifest.yaml`，内容应与 `manifest.yaml` 一致。修改权重路径、数据集 ID、下载命令、运行命令或配置适配说明后，必须同步更新两个 Manifest 并重新执行 YAML 解析。

## 模型 vs 数据集关系

模型 `OneScience/protenix/` 必需数据集为 `OneScience/protenix_dataset`。模型 Manifest 的 `relations.required_datasets` 提供完整 `resource_ref`，数据集 Manifest 的 `relations.compatible_models` 反向指向本模型。推荐下载布局：

```text
workspace/
├── bio_protenix/
└── bio_protenix_dataset/
```

默认 `DATA_ROOT_DIR=../bio_protenix_dataset`。若网页端使用 `--cache_dir` 下载模型，运行 `cwd` 必须切换到实际下载后的模型包根目录。

## 文件与下载

下载模型：

```bash
modelscope download --model OneScience/protenix/ --local_dir ./bio_protenix
```

下载数据集：

```bash
modelscope download --dataset OneScience/protenix_dataset --local_dir ./bio_protenix_dataset
```

## 环境安装

如果 OneScience bio 环境尚不可用，执行：

```bash
bash install.sh bio
```

## 运行流程

### 1. 环境预检

```bash
cd ./bio_protenix
python scripts/preflight_protenix.py --model-root . --data-root ../bio_protenix_dataset
```

### 2. 下载

```bash
modelscope download --model OneScience/protenix/ --local_dir ./bio_protenix
modelscope download --dataset OneScience/protenix_dataset --local_dir ./bio_protenix_dataset
```

### 3. 应用运行包和准备文件

```bash
cd ./bio_protenix
export DATA_ROOT_DIR=../bio_protenix_dataset
```

### 4. 运行前预检

```bash
python scripts/preflight_protenix.py --model-root . --data-root "$DATA_ROOT_DIR"
```

### 5. 运行

```bash
bash inference_unified_demo.sh
```

训练或微调：

```bash
bash train_demo.sh
bash finetune_demo.sh
```

### 6. 验证输出

推理成功后应生成 `output_unified/7r6r/seed_101/predictions/*.cif` 和 `*summary_confidence*.json`。训练或微调输出位于 `output/`。

## 预检与诊断

| 现象 | 常见原因 | 处理方式 |
|---|---|---|
| `No such file or directory` | 模型权重、样例输入或数据集目录缺失 | 重新执行模型和数据集下载命令，并运行预检 |
| `KeyError: 'DATA_ROOT_DIR'` | 数据根目录环境变量未设置 | `export DATA_ROOT_DIR=../bio_protenix_dataset` |
| `ModuleNotFoundError` | OneScience bio 环境或依赖缺失 | 执行 `bash install.sh bio` |
| `CUDA out of memory` | 采样数、扩散步数或 batch 过大 | 降低 `N_sample`、`N_step` 或 `diffusion_batch_size` |
| 找不到 CCD 或 RDKit 缓存 | 数据集未放在 Manifest 声明路径 | 检查 `../bio_protenix_dataset/components.v20240608.cif*` |

## 输出说明

推理输出目录为 `output_unified/7r6r/seed_101/predictions/`。CIF 文件是结构预测结果，JSON 文件是样本置信度摘要。训练和微调输出目录为 `output/`，包含日志、检查点和中间结果。

## 限制与适用范围

本包面向 Protenix 在 OneScience 中的推理、训练和微调，不提供独立部署服务。完整训练和微调需要完整 Protenix 数据集、较大磁盘空间以及 GPU/DCU 资源。

## 引用与许可证

请遵守 Protenix、OneScience、wwPDB、CCD、recentPDB、posebusters 及相关 MSA 数据源的许可证和引用要求。本仓库整理目标是支持 `OneScience/protenix/` 在 ModelScope 和 OneScience 网站端自动运行。
