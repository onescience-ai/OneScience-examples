# NEP

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 NEP 模型训练预检</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/NEP" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

NEP 是 OneScience MatChem 领域基于 MatPL 的神经网络势训练示例，支持 Cu、LiSiC、AuAg、HfO2 等体系。本标准模型包整理了 `onescience/examples/matchem/nep` 中的安装脚本、示例 JSON 和提交脚本，并新增面向 ModelScope 数据布局的 Cu 入门配置。

原始 Cu 配置使用 `/public/share/.../matchem/matpl/...` 绝对路径。本标准包新增 `conf/Cu_nep_train_modelscope.json`，将数据路径改为下载后的相对路径 `data/MatPL/Cu/pwdata/...`。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | matchem |
| 领域标签 | matchem, nep, matpl, mlip |
| 任务 | neural_evolution_potential_training |
| 任务标签 | energy_prediction, force_prediction, virial_prediction, movement |
| 主平台资源 | https://modelscope.cn/models/OneScience/NEP |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/matchem/nep` |
| 支持能力 | 训练 / 评测 / 预检 |
| 必需数据集 | `OneScience/MatPL` |
| 最小验证 | `python scripts/preflight_nep.py --package-root .` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 说明文档 | 模型用途、文件、下载、运行和诊断说明 | 是 | 全部能力 | `session_workdir/README.md` | 本文件 |
| `manifest.yaml` | Manifest 文件 | 标准默认机器可读运行说明 | 是 | 全部能力 | `session_workdir/manifest.yaml` | 与 `onescience_run_manifest.yaml` 一致 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 大模型运行 Manifest 文件名 | 是 | 全部能力 | `session_workdir/onescience_run_manifest.yaml` | 修改后需同步 |
| `scripts/preflight_nep.py` | 预检脚本 | 检查运行文件、Cu 配置和 MatPL 数据路径 | 是 | 预检 | `session_workdir/scripts/preflight_nep.py` | 不启动训练 |
| `conf/Cu_nep_train_modelscope.json` | 适配配置 | 面向 ModelScope 下载布局的 NEP Cu 训练配置 | 是 | 训练、预检 | `session_workdir/conf/Cu_nep_train_modelscope.json` | 已改数据路径 |
| `upstream/` | 上游示例代码 | NEP 原始安装、JSON 配置和提交脚本 | 是 | 训练、说明 | `session_workdir/upstream/` | 来自 OneScience 示例 |
| `metadata/sha256_manifest.txt` | 校验清单 | 模型包脚本和配置 SHA256 | 是 | 预检 | `session_workdir/metadata/sha256_manifest.txt` | 上传前校验 |

## Manifest

本仓库提供 `manifest.yaml` 和 `onescience_run_manifest.yaml`，两者内容一致。修改文件、下载命令、数据关系或适配配置后必须同步更新。

## 模型 vs 数据集关系

模型仓库目标 ID 是 `OneScience/NEP`，数据集仓库目标 ID 是 `OneScience/MatPL`。模型 Manifest 的 `relations.required_datasets` 指向 `OneScience/MatPL`，数据集 Manifest 的 `relations.compatible_models` 反向指向 `OneScience/NEP`。

## 文件与下载

```bash
modelscope download --model OneScience/NEP --local_dir session_workdir
modelscope download --dataset OneScience/MatPL --local_dir session_workdir
```

如果网页端使用 `--cache_dir` 下载模型，运行前必须切换到实际下载后的模型包根目录。

## 环境安装

```bash
bash install.sh matchem
cd upstream
bash matpl_install.sh
```

## 运行流程

### 1. 下载

```bash
modelscope download --model OneScience/NEP --local_dir session_workdir
modelscope download --dataset OneScience/MatPL --local_dir session_workdir
cd session_workdir
```

### 2. 运行前预检

```bash
python scripts/preflight_nep.py --package-root .
```

### 3. 训练

```bash
MatPL train conf/Cu_nep_train_modelscope.json
```

如果使用 SLURM，可参考 `upstream/demo/nep_Cu/submit.sh`，并将末尾训练命令替换为上述标准配置路径。

## 输出说明

MatPL 会在当前运行目录输出训练日志、NEP 模型文件和中间结果；SLURM 模式下还会产生 `slurm_*.out` 和 `slurm_*.err`。

## 预检与诊断

- `MatPL: command not found`：MatPL 未安装或环境未加载。
- `missing file`：未下载 `OneScience/MatPL` 或数据未放在 `data/MatPL/`。
- `train_data mismatch`：配置文件不是标准适配后的 `conf/Cu_nep_train_modelscope.json`。
- `ModuleNotFoundError`：缺少 Python 依赖，例如 numpy。

## 限制与适用范围

本标准包默认使用 Cu 入门配置做最小验证和训练入口。AuAg、HfO2、LiSiC 的原始示例配置保留在 `upstream/demo/`，如需使用这些配置，应按相同规则把绝对路径改为 `data/MatPL/...`。

## 引用与许可证

NEP 示例代码来自 OneScience 仓库，许可证以 OneScience 仓库说明为准。MatPL 数据使用限制以原始数据来源和 ModelScope 页面说明为准。
