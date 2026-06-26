# AlphaFold3

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 AlphaFold3 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/AlphaFold3/" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

AlphaFold3 是面向生物分子结构预测的模型运行包，可接收 AlphaFold3 JSON 输入，输出结构预测结果、ranking scores 和中间数据管线结果。本仓库整理为 OneScience 标准运行包，模型 ID 固定为 `OneScience/AlphaFold3/`，需要的数据集 ID 固定为 `OneScience/AlphaFold3_dataset`。

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

AlphaFold3 用于蛋白质、核酸、配体等生物分子体系的结构预测。它通常读取 AlphaFold3 JSON，输入中可以已经包含 MSA，也可以只包含序列并通过数据库搜索生成 MSA。输出包括结构文件、处理后的 fold input JSON 和 ranking score 等结果。

本仓库提供 OneScience 兼容的 `run_alphafold.py`、推理包装脚本、路径配置和预检脚本。模型权重、运行库和 mmseqs 二进制位于 `checkpoints/AlphaFold3/`，数据集通过 `OneScience/AlphaFold3_dataset` 下载后放到 `data/alphafold3_dataset/`。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | bio |
| 领域标签 | bio, biosciences, protein_structure, molecular_structure |
| 任务 | biomolecular_structure_prediction |
| 任务标签 | inference, msa_search, data_pipeline, structure_prediction |
| 主平台资源 | https://modelscope.cn/models/OneScience/AlphaFold3/ |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/biosciences/alphafold3` |
| 必需模型文件 | `checkpoints/AlphaFold3/af3.bin`, `libflash_atten_c.so`, `mmseqs/` |
| 必需数据集 | `OneScience/AlphaFold3_dataset` |
| 支持能力 | 预检、带 MSA 推理、MMseqs/Jackhmmer 数据管线 |
| 最小验证 | `python scripts/preflight.py --root . --dataset-root data/alphafold3_dataset --model-dir checkpoints/AlphaFold3` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、关系、命令和输出 | 是 | 全部能力 | 模型包根目录 | 大模型必须解析 |
| `onescience_run_manifest.yaml` | 运行 Manifest | 与 `manifest.yaml` 保持一致，供网页端直接读取 | 是 | 全部能力 | 模型包根目录 | 包含配置适配说明 |
| `conf/alphafold3_paths.yaml` | 配置文件 | 记录权重、数据集、输入和输出默认路径 | 是 | 预检、推理、数据管线 | `conf/` | 已从原始环境变量路径适配到包内路径 |
| `run_alphafold.py` | 运行脚本 | OneScience AlphaFold3 主入口 | 是 | 推理、数据管线 | 模型包根目录 | 来自 OneScience 示例 |
| `run_inference_msa.sh` | 包装脚本 | 使用带 MSA 的 `7r6r_data.json` 执行推理 | 是 | 推理 | 模型包根目录 | 需要 GPU 和 OneScience bio 环境 |
| `run_data_pipeline_mmseqs.sh` | 包装脚本 | 使用 MMseqs 数据库执行搜索数据管线 | 是 | 数据管线 | 模型包根目录 | 默认不跑推理 |
| `run_data_pipeline_jackhmmer.sh` | 包装脚本 | 使用 Jackhmmer 分片数据库执行搜索数据管线 | 是 | 数据管线 | 模型包根目录 | 默认不跑推理 |
| `scripts/preflight.py` | 预检脚本 | 检查配置、权重、数据路径、JSON schema 和关键文件哈希 | 是 | 预检 | `scripts/` | 不加载模型到 GPU |
| `checkpoints/AlphaFold3/` | 模型文件 | 权重、动态库、mmseqs 可执行文件 | 是 | 推理、数据管线 | `checkpoints/AlphaFold3/` | 本地整理为链接 |
| `inputs/7r6r_data.json` | 样例输入 | 已带 MSA 的推理样例 | 是 | 最小推理 | `inputs/` | 可跳过数据库搜索 |
| `inputs/t1119_search.json` | 样例输入 | 无 MSA 的搜索样例 | 是 | 数据管线 | `inputs/` | 配合数据集数据库使用 |

## Manifest

独立 Manifest 文件位于仓库根目录：`manifest.yaml`。网页端也可以读取同目录下的 `onescience_run_manifest.yaml`，两者资源 ID、relations、run_matrix 和配置适配说明保持一致。修改任何运行脚本、文件路径、下载命令或数据集关系时，必须同步更新这两个 Manifest。

## 模型 vs 数据集关系

本模型必须与数据集 `OneScience/AlphaFold3_dataset` 配套使用。模型 Manifest 的 `relations.required_datasets` 指向完整的 `resource_ref`：`https://modelscope.cn/datasets/OneScience/AlphaFold3_dataset`。数据集 Manifest 的 `relations.compatible_models` 反向指向 `OneScience/AlphaFold3/`。

带 MSA 的最小推理场景只需要模型权重和 `inputs/7r6r_data.json`。如果需要从序列搜索 MSA，则必须下载数据集并放到 `data/alphafold3_dataset/`，其中 `mmseqsDB` 支持 MMseqs 流程，`jackhmmer_split` 和 `public_databases` 支持 Jackhmmer 流程。

## 文件与下载

下载模型：

```bash
modelscope download --model OneScience/AlphaFold3/
```

下载数据集：

```bash
modelscope download --dataset OneScience/AlphaFold3_dataset
```

如果使用 `--cache_dir`，运行前必须切换到实际下载出的模型包根目录。数据集下载到独立目录时，应将数据集包中的 `data` 子目录放置或链接到：

```bash
mkdir -p data
ln -s /path/to/AlphaFold3_dataset/data data/alphafold3_dataset
```

## 环境安装

默认假设 OneScience 网站环境已安装 bio 领域依赖。环境缺失时参考 OneScience 官方仓库安装：

```bash
bash install.sh bio
```

推理需要可用 GPU、JAX 后端和 AlphaFold3 相关依赖。预检脚本只检查文件和 schema，不要求 GPU。

## 运行流程

先执行预检：

```bash
python scripts/preflight.py --root . --dataset-root data/alphafold3_dataset --model-dir checkpoints/AlphaFold3
```

使用带 MSA 输入进行最小推理：

```bash
bash run_inference_msa.sh inputs/7r6r_data.json outputs/msa_inference
```

使用 MMseqs 数据管线：

```bash
bash run_data_pipeline_mmseqs.sh inputs/t1119_search.json outputs/mmseqs_pipeline
```

使用 Jackhmmer 数据管线：

```bash
bash run_data_pipeline_jackhmmer.sh inputs/t1119_search.json outputs/jackhmmer_pipeline
```

## 预检与诊断

常见错误与处理：

| 错误现象 | 可能原因 | 处理方式 |
|---|---|---|
| `missing_af3_bin` | 模型未完整下载或 cwd 不是模型包根目录 | 重新执行 `modelscope download --model OneScience/AlphaFold3/`，再切换到模型包根目录 |
| `missing_dataset_database` | 数据集未放到 `data/alphafold3_dataset` | 下载 `OneScience/AlphaFold3_dataset` 并建立链接 |
| `sha256 mismatch` | 链接或复制的数据与原始文件不一致 | 对照 `metadata_model_files.yaml` 或数据集 `metadata/file_manifest.yaml` 重新整理 |
| `jax_gpu_not_found` | 无 GPU 或 JAX 后端不可用 | 先执行预检，推理切换到具备 GPU 的 OneScience bio 环境 |
| `mmseqs not found` | PATH 未包含 mmseqs | 使用 `run_data_pipeline_mmseqs.sh` 包装脚本 |

## 输出说明

预检成功时输出 `model_preflight_ok: true`。最小推理输出位于 `outputs/msa_inference/`，通常包含每个 fold job 的结构结果、处理后的 JSON 和 ranking score CSV。数据管线输出位于 `outputs/mmseqs_pipeline/` 或 `outputs/jackhmmer_pipeline/`。

## 限制与适用范围

当前标准包不声明训练、微调、评测或部署能力，因为原始 `run_alphafold3_trainer.py` 为空，且本次资源主要包含推理权重和搜索数据库。数据库规模很大，本地整理目录采用符号链接指向原始只读目录；正式上传前可按平台要求改为实体文件或大文件存储策略，并重新运行校验。

## 引用与许可证

AlphaFold3 源码和权重使用条款需遵守上游 Google DeepMind AlphaFold3 许可与权重使用条款。OneScience 代码、脚本和包装说明遵守 OneScience 项目对应许可证。公共数据库遵守各自上游数据源许可。
