# MACE

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 MACE 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/MACE" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

MACE 是面向分子和材料体系的机器学习原子间势模型，常用于预测体系能量、原子受力以及基于原子结构的训练和评测流程。本标准仓库整理的是 ANI-1x 训练场景的可运行包，用户下载后只需要把数据集放到模型包根目录的 `data/ani1x/`，再执行标准化配置即可完成预检、训练和基于验证集的评测。

这个包的输入是分子构型的 HDF5/XYZ 数据和统计文件，输出是训练日志、checkpoint 和评测过程中的验证结果。它适合做最小可运行验证、分布式训练预检、ANI-1x 数据集兼容性检查，以及后续继续训练或评测。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | matchem |
| 领域标签 | matchem, molecular_potential, atomistic_simulation |
| 任务 | mlip_training |
| 任务标签 | mlip, energy_prediction, force_prediction, distributed_training |
| 主平台资源 | https://modelscope.cn/models/OneScience/MACE |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/matchem/mace` |
| 支持能力 | 训练 / 评测 |
| 必需模型文件 | `train.py`, `run.sh`, `_parse_config.py`, `configs/ani1x_8dcu.yaml`, `scripts/preflight_mace_ani1x.py` |
| 必需数据集 | `OneScience/ani1x` |
| 最小验证 | `python scripts/preflight_mace_ani1x.py --config configs/ani1x_8dcu.yaml --data-dir data/ani1x` |

能力和命令要求
| 能力 | 必须提供 |
|---|---|
| `train` | 训练命令、训练数据、配置文件、输出 checkpoint 路径 |
| `evaluate` | 评测命令、评测数据、指标、结果路径 |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源身份、文件、关系、命令和输出。大模型必须先读 README 的文件说明再打开该文件 | 是 | 全部能力 | `session_workdir/manifest.yaml` | 修改文件路径、下载方式或命令后必须同步更新 |
| `README.md` | 说明文档 | 资源用途、下载、运行和诊断说明 | 是 | 全部能力 | `session_workdir/README.md` | 本文件 |
| `train.py` | 运行脚本 | MACE 训练主入口 | 是 | 训练、评测 | `session_workdir/train.py` | 来自 OneScience examples 兼容实现 |
| `run.sh` | 运行脚本 | 统一训练入口，支持 dry-run、直接运行和 SLURM 提交 | 是 | 训练、评测 | `session_workdir/run.sh` | 通过 `configs/ani1x_8dcu.yaml` 驱动 |
| `_parse_config.py` | 运行脚本 | 解析 YAML 配置并生成训练命令和预检文件列表 | 是 | 训练、预检 | `session_workdir/_parse_config.py` | 已适配标准包目录 |
| `configs/ani1x_8dcu.yaml` | 配置文件 | ANI-1x 8 DCU 训练配置 | 是 | 训练、评测 | `session_workdir/configs/ani1x_8dcu.yaml` | 已改为读取 `data/ani1x/` |
| `metadata/sha256_manifest.txt` | 校验清单 | `data/ani1x/` 中数据文件的 SHA256 清单 | 是 | 预检、上传校验 | `session_workdir/metadata/sha256_manifest.txt` | 用于确认模型包内复制数据与原始数据一致 |
| `scripts/preflight_mace_ani1x.py` | 预检脚本 | 检查配置、统计文件、HDF5 字段、shape、dtype、统计值一致性 | 是 | 预检 | `session_workdir/scripts/preflight_mace_ani1x.py` | 上传前和运行前都可执行 |
| `data/ani1x/` | 数据目录 | ANI-1x 训练、验证、测试 HDF5 分片、statistics 和 XYZ 辅助文件 | 是 | 预检、训练、评测 | `session_workdir/data/ani1x/` | 为便于模型仓库独立上传和预检，此处为普通文件拷贝，不是软链接 |

## Manifest

独立 Manifest 文件位于仓库根目录的 `manifest.yaml`。如果修改了模型文件、数据路径、下载命令或 `configs/ani1x_8dcu.yaml`，必须同步更新该文件中的 `files`、`relations`、`run_matrix` 和 `configuration_adaptation`。

建议在上传前执行：

```bash
python - <<'PY'
import yaml
yaml.safe_load(open('manifest.yaml', encoding='utf-8'))
print('YAML OK')
PY
```

## 模型 vs 数据集关系

该模型的标准训练与评测都依赖 `OneScience/ani1x`。模型仓库通过 `relations.required_datasets` 显式声明这个依赖，数据集仓库会反向声明其兼容模型。网页端或大模型收到这个模型后，应先下载模型包，再下载 `OneScience/ani1x`，并把数据放到 `data/ani1x/`。

`run_matrix` 中的最小验证场景只做预检，训练场景则需要训练分片、验证分片和统计文件。当前包没有独立推理入口，因为该整理目标是训练/评测标准包，不是推理服务包。

## 文件与下载

本模型仓库已经包含一份普通文件拷贝的 `data/ani1x/`，可直接用于预检和训练。若需要从独立数据集仓库重新获取数据，下载命令如下：

```bash
modelscope download --model OneScience/MACE --local_dir session_workdir
modelscope download --dataset OneScience/ani1x --local_dir session_workdir
```

下载后将数据集内容保留在 `session_workdir/data/ani1x/`，模型包根目录保持为运行 cwd。网页端如果使用 `--cache_dir` 下载模型，运行前必须切换到实际解压后的模型包根目录，再执行 `bash run.sh --config configs/ani1x_8dcu.yaml`。

## 环境安装

```bash
bash install.sh matchem
```

## 运行流程

### 1. 环境预检

```bash
python scripts/preflight_mace_ani1x.py --config configs/ani1x_8dcu.yaml --data-dir data/ani1x
```

### 2. 下载

```bash
modelscope download --model OneScience/MACE --local_dir session_workdir
modelscope download --dataset OneScience/ani1x --local_dir session_workdir
```

### 3. 应用运行包和准备文件

```bash
mkdir -p data
cp -a /path/to/downloaded/ani1x data/ani1x
```

### 4. 运行前预检

```bash
python scripts/preflight_mace_ani1x.py --config configs/ani1x_8dcu.yaml --data-dir data/ani1x
```

### 5. 运行

```bash
bash run.sh --config configs/ani1x_8dcu.yaml
```

### 6. 验证输出

训练结束后检查 `outputs/` 目录是否生成实验子目录，并确认日志中出现验证指标和 checkpoint 保存信息。最小验证只要求预检脚本返回 `OK`，说明配置和数据可读。

## 输出说明

主要输出位于 `outputs/{实验名}_{时间戳}/`，通常包含：

- 训练日志
- `config.yaml` 快照
- checkpoint 文件
- 验证过程中的错误统计和训练曲线

如果只做最小验证，输出主要是标准输出中的 `[OK]` 检查结果。

## 预检与诊断

常见问题如下：

- `No such file or directory`：模型包文件或 `data/ani1x` 不完整。
- `r_max` 相关报错：配置里的 `r_max` 与 statistics 文件不一致。
- `ModuleNotFoundError`：OneScience matchem 依赖未安装。
- `CUDA out of memory`：batch size 过大，需要降小 `batch_size` 或 `num_channels`。

## 限制与适用范围

这个包主要面向 ANI-1x 训练和验证流程，不包含独立推理服务、部署脚本或可视化页面。它依赖的输入数据必须是 MACE 可读的 HDF5/XYZ 结构，并且 statistics 文件要和训练配置保持一致。

## 引用与许可证

MACE 相关代码来自 OneScience 项目中的 matchem 示例实现。许可证以 OneScience 主仓库和上游 MACE 许可为准。
