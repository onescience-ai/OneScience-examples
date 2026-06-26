# UMA

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 UMA 模型训练预检</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/UMA" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

UMA 是 OneScience MatChem 领域的通用材料相互作用模型示例，当前标准包整理了 `onescience/examples/matchem/uma` 中的训练、推理、数据转换脚本和 OC20 微调配置。模型目标仓库 ID 为 `OneScience/UMA`，适配数据集为 `OneScience/oc20`。

原始示例依赖环境变量 `ONESCIENCE_DATASETS_DIR` 和 `ONESCIENCE_MODELS_DIR`。本标准包新增 `conf/oc20_ef_4dcu_modelscope.yaml`，将 OC20 数据路径改为下载后的相对路径 `data/oc20/uma_oc20_finetune/{train,val}`，并将 checkpoint 路径声明为 `checkpoints/uma-s-1p1_converted.pt`。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | matchem |
| 领域标签 | matchem, uma, mlip, catalysis |
| 任务 | oc20_energy_force_finetuning |
| 任务标签 | oc20, energy_prediction, force_prediction, finetune |
| 主平台资源 | https://modelscope.cn/models/OneScience/UMA |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/matchem/uma` |
| 支持能力 | 训练 / 评测 / 推理脚本 / 预检 |
| 必需数据集 | `OneScience/oc20` |
| 最小验证 | `python scripts/preflight_uma.py --package-root .` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 说明文档 | 模型用途、文件、下载、运行和诊断说明 | 是 | 全部能力 | `session_workdir/README.md` | 本文件 |
| `manifest.yaml` | Manifest 文件 | 标准默认机器可读运行说明 | 是 | 全部能力 | `session_workdir/manifest.yaml` | 与 `onescience_run_manifest.yaml` 一致 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 大模型运行 Manifest 文件名 | 是 | 全部能力 | `session_workdir/onescience_run_manifest.yaml` | 修改后需同步 |
| `scripts/preflight_uma.py` | 预检脚本 | 检查运行文件、配置、OC20 数据路径和可选 checkpoint | 是 | 预检 | `session_workdir/scripts/preflight_uma.py` | 不启动训练 |
| `conf/oc20_ef_4dcu_modelscope.yaml` | 适配配置 | 面向 ModelScope 下载布局的 UMA OC20 微调配置 | 是 | 训练、预检 | `session_workdir/conf/oc20_ef_4dcu_modelscope.yaml` | 已改数据路径 |
| `upstream/` | 上游示例代码 | UMA 原始训练、推理、转换脚本和 demo 配置 | 是 | 训练、推理、转换 | `session_workdir/upstream/` | 来自 OneScience 示例 |
| `metadata/sha256_manifest.txt` | 校验清单 | 模型包脚本和配置 SHA256 | 是 | 预检 | `session_workdir/metadata/sha256_manifest.txt` | 上传前校验 |
| `checkpoints/uma-s-1p1_converted.pt` | checkpoint | UMA 微调所需权重 | 否 | 训练 | `session_workdir/checkpoints/` | 当前本地源目录未提供权重 |

## Manifest

本仓库提供 `manifest.yaml` 和 `onescience_run_manifest.yaml`，两者内容一致。修改文件、下载命令、数据关系或适配配置后必须同步更新。

## 模型 vs 数据集关系

模型仓库目标 ID 是 `OneScience/UMA`，数据集仓库目标 ID 是 `OneScience/oc20`。模型 Manifest 的 `relations.required_datasets` 指向 `OneScience/oc20`，数据集 Manifest 的 `relations.compatible_models` 反向指向 `OneScience/UMA`。

## 文件与下载

```bash
modelscope download --model OneScience/UMA --local_dir session_workdir
modelscope download --dataset OneScience/oc20 --local_dir session_workdir
```

如果网页端使用 `--cache_dir` 下载模型，运行前必须切换到实际下载后的模型包根目录。

## 环境安装

```bash
bash install.sh matchem
```

## 运行流程

### 1. 下载

```bash
modelscope download --model OneScience/UMA --local_dir session_workdir
modelscope download --dataset OneScience/oc20 --local_dir session_workdir
cd session_workdir
```

### 2. 运行前预检

```bash
python scripts/preflight_uma.py --package-root .
```

如果已经放置 checkpoint，可加严格检查：

```bash
python scripts/preflight_uma.py --package-root . --require-checkpoint
```

### 3. 训练 dry-run

```bash
cd upstream/demo
bash run.sh --config ../../conf/oc20_ef_4dcu_modelscope.yaml --dry-run
```

### 4. 训练

```bash
cd upstream/demo
bash run.sh --config ../../conf/oc20_ef_4dcu_modelscope.yaml
```

## 输出说明

`run.sh` 会在 `upstream/demo/outputs/` 下生成配置快照、Hydra 配置、训练日志和 `uma_finetune_runs/` 目录。dry-run 只打印命令和配置预览，不启动训练。

## 预检与诊断

- `missing directory`：未下载 `OneScience/oc20` 或数据没有放在 `data/oc20/`。
- `checkpoint not present`：当前本地源目录未提供 UMA checkpoint，请将转换后的权重放到 `checkpoints/uma-s-1p1_converted.pt`。
- `ModuleNotFoundError`：缺少 OneScience matchem、torch、fairchem/ASE 等依赖。
- `配置文件不存在`：运行 `run.sh` 时 cwd 或 `--config` 路径错误。

## 限制与适用范围

本标准包整理 UMA 示例代码和 OC20 微调配置；当前本地源目录没有提供 checkpoint 文件，因此上传前如需一键真实训练，应补充 `checkpoints/uma-s-1p1_converted.pt` 或在 README/Manifest 中保持 checkpoint 外部依赖说明。

## 引用与许可证

UMA 示例代码来自 OneScience 仓库，许可证以 OneScience 仓库说明为准。OC20 数据使用限制以原始数据来源和 ModelScope 页面说明为准。
