# DeePMD

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 DeePMD 模型训练预检</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/DeePMD" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

DeePMD 是 OneScience MatChem 领域的 DeepMD-kit 训练示例，包含 PyTorch 后端、TensorFlow 后端以及单卡/多卡 SLURM 提交脚本。本标准模型包整理了 `onescience/examples/matchem/dp`，并新增面向 ModelScope 下载布局的 PyTorch water 配置。

原始 PyTorch 配置使用 `/public/share/.../matchem/dp/water/data_*` 绝对路径。本标准包新增 `conf/input_torch_modelscope.json`，将数据路径改为 `data/DeePMD/water/data_0..3`。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | matchem |
| 领域标签 | matchem, deepmd, water, mlip |
| 任务 | deepmd_water_training |
| 任务标签 | energy_prediction, force_prediction, pytorch_backend |
| 主平台资源 | https://modelscope.cn/models/OneScience/DeePMD |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/matchem/dp` |
| 支持能力 | 训练 / 评测 / 预检 |
| 必需数据集 | `OneScience/DeePMD` |
| 最小验证 | `python scripts/preflight_deepmd.py --package-root .` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 说明文档 | 模型用途、文件、下载、运行和诊断说明 | 是 | 全部能力 | `session_workdir/README.md` | 本文件 |
| `manifest.yaml` | Manifest 文件 | 标准默认机器可读运行说明 | 是 | 全部能力 | `session_workdir/manifest.yaml` | 与 `onescience_run_manifest.yaml` 一致 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 大模型运行 Manifest 文件名 | 是 | 全部能力 | `session_workdir/onescience_run_manifest.yaml` | 修改后需同步 |
| `scripts/preflight_deepmd.py` | 预检脚本 | 检查运行文件、配置和 water 数据路径 | 是 | 预检 | `session_workdir/scripts/preflight_deepmd.py` | 不启动训练 |
| `conf/input_torch_modelscope.json` | 适配配置 | 面向 ModelScope 下载布局的 PyTorch water 训练配置 | 是 | 训练、预检 | `session_workdir/conf/input_torch_modelscope.json` | 已改数据路径 |
| `upstream/` | 上游示例代码 | DeePMD 原始安装、配置和提交脚本 | 是 | 训练、说明 | `session_workdir/upstream/` | 来自 OneScience 示例 |
| `metadata/sha256_manifest.txt` | 校验清单 | 模型包脚本和配置 SHA256 | 是 | 预检 | `session_workdir/metadata/sha256_manifest.txt` | 上传前校验 |

## Manifest

本仓库提供 `manifest.yaml` 和 `onescience_run_manifest.yaml`，两者内容一致。修改文件、下载命令、数据关系或适配配置后必须同步更新。

## 模型 vs 数据集关系

模型仓库和数据集仓库的目标 ID 都是 `OneScience/DeePMD`，但 repo_type 不同：模型仓库使用 `repo_type: model`，数据集仓库使用 `repo_type: dataset`。模型 Manifest 的 `relations.required_datasets` 指向 `OneScience/DeePMD` 数据集，并提供完整 `resource_ref`。

## 文件与下载

```bash
modelscope download --model OneScience/DeePMD --local_dir session_workdir
modelscope download --dataset OneScience/DeePMD --local_dir session_workdir
```

如果网页端使用 `--cache_dir` 下载模型，运行前必须切换到实际下载后的模型包根目录。

## 环境安装

```bash
bash install.sh matchem
cd upstream
bash dp_install.sh
```

## 运行流程

### 1. 下载

```bash
modelscope download --model OneScience/DeePMD --local_dir session_workdir
modelscope download --dataset OneScience/DeePMD --local_dir session_workdir
cd session_workdir
```

### 2. 运行前预检

```bash
python scripts/preflight_deepmd.py --package-root .
```

### 3. 训练

```bash
dp --pt train conf/input_torch_modelscope.json
```

多卡训练可参考 `upstream/demo/water_se_e2_a_pt/submit_4card.sh` 和 `submit_8card.sh`，并将训练配置替换为 `conf/input_torch_modelscope.json`。

## 输出说明

DeepMD-kit 训练会输出 `lcurve.out`、统计文件、checkpoint 和日志。SLURM 模式下还会产生 `slurm_*.out` 和 `slurm_*.err`。

## 预检与诊断

- `dp: command not found`：DeepMD-kit 未安装或环境未加载。
- `missing directory`：未下载 `OneScience/DeePMD` 数据集或数据未放在 `data/DeePMD/`。
- `training systems mismatch`：配置文件不是标准适配后的 `conf/input_torch_modelscope.json`。
- `ModuleNotFoundError`：缺少 Python 依赖，例如 numpy。

## 限制与适用范围

本标准包默认使用 PyTorch `se_e2_a` water 配置做最小验证和训练入口。attention 与 TensorFlow 示例保留在 `upstream/demo/`，如需使用这些配置，应按相同规则把绝对路径改为 `data/DeePMD/...`。

## 引用与许可证

DeePMD 示例代码来自 OneScience 仓库，DeepMD-kit 相关许可请参考其上游项目。数据使用限制以 OneScience 仓库和 ModelScope 页面说明为准。
