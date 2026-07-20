# UMA

UMA（Universal Materials Interaction Model）是面向材料与催化体系的通用机器学习原子间势示例模型，基于等变图神经网络构建，可用于原子结构的能量、受力预测，并支持 OC20、OC22、OC25、OMat、OMOL、ODAC、OMC 等多种材料与催化任务的微调训练与推理。

本仓库中的模型实现来自 OneScience MatChem 领域。

---

## 仓库说明

本仓库是 OneScience 整理的 UMA 最小可运行模型仓库，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 多任务能量 + 受力微调训练：OC20、OC22、OC25、OMat、OMOL、ODAC、OMC
- 训练 dry-run：生成训练命令和 Hydra 配置预览，不启动真实训练
- 预检：检查运行文件、配置、数据路径和可选 checkpoint 是否齐全
- 推理脚本参考：无机晶体弛豫、吸附体系弛豫、分子 MD 和批量推理示例
- GPU/DCU 优先运行

当前不支持能力：

- 不内置 UMA checkpoint，真实微调前需准备权重文件
- 不内置各任务真实训练数据，需自行准备
- 不提供独立在线推理服务、部署脚本或可视化页面

---

## 适用场景

| 场景 | 说明 |
| :---: | :---: |
| OC20 能量和力微调 | 使用标准配置读取 OC20 微调数据并训练 UMA 模型 |
| OC22 氧化物催化微调 | 使用标准配置读取 OC22 微调数据并训练 UMA 模型（仅限 1P2） |
| OC25 （电）催化微调 | 使用标准配置读取 OC25 微调数据并训练 UMA 模型（仅 1P2） |
| OMat 无机材料微调 | 使用标准配置读取 OMat 微调数据并训练 UMA 模型 |
| OMOL 分子+聚合物微调 | 使用标准配置读取 OMOL 微调数据并训练 UMA 模型 |
| ODAC MOFs 微调 | 使用标准配置读取 ODAC 微调数据并训练 UMA 模型 |
| OMC 分子晶体微调 | 使用标准配置读取 OMC 微调数据并训练 UMA 模型 |
| 训练流程预检 | 检查配置、数据路径、运行脚本和 checkpoint 放置位置 |
| 催化吸附体系建模 | 参考 OC20/OC22/OC25 任务进行吸附/催化表面体系训练与推理 |
| 推理脚本参考 | 使用上游推理示例进行晶体弛豫、吸附体系弛豫或分子 MD 改造 |
| 自有数据迁移 | 将 ASE 可读结构转换为 UMA 微调数据后替换训练和验证路径 |

---

## 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 本文件 |
| `model/` | UMA 模型源码 | 包含 `__init__.py`、`base.py`、`uma_escn_md.py`、`uma_escn_moe.py`、`models/` |
| `train.py` | UMA 训练主入口 | 来自 OneScience matchem 示例 |
| `demo/run.sh` | 统一训练入口 | 支持直接运行、dry-run 和 SLURM 提交 |
| `demo/_parse_config.py` | 配置解析脚本 | 生成训练命令、Hydra 配置和预检文件列表 |
| `demo/configs/` | 训练配置文件 | 包含 `oc20_ef_4dcu.yaml` 等示例配置；可按需添加 OC22、OC25、OMat、OMOL、ODAC、OMC 等任务配置 |
| `demo/templates/` | 脚本模板 | 环境初始化、预检、SLURM header 模板 |
| `configs/` | UMA 数据/任务配置模板 | 包含 `uma_sm_finetune_template.yaml`、`configs/data/` |
| `scripts/` | 数据转换和模型转换脚本 | 自定义微调数据集、checkpoint 转换、demo 配置更新 |
| `scripts/update_demo_config.py` | 更新 demo 配置文件 | 把 `create_uma_finetune_dataset.py` 生成的 data yaml 同步到 demo config |
| `inference/` | UMA 推理示例脚本 | 晶体弛豫、吸附体系、分子 MD、批量推理 |
| `models-json/` | 预训练模型清单 | 训练/推理时通过输出目录的 `models` 软链访问 |

---

## 使用说明

### 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

### 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行微调训练。
- CPU 可以用于配置和数据路径预检，不建议用于正式 UMA 训练。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**软件要求**

- Python 3.11
- OneScience matchem 运行环境

安装运行环境：

DCU环境

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

GPU环境

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 3. 快速开始

**进入示例目录**

本示例位于 OneScience Examples 的 `models/matchem/UMA`，进入该目录后所有命令均相对于该目录执行：

```bash
cd models/matchem/UMA
```

**准备数据**

本仓库不内置训练数据。以下以 **OC20 微调**为例说明流程，OC22、OC25、OMat、OMOL、ODAC、OMC 等其他任务流程相同，只需替换 `--uma-task` 和数据路径即可。

方式一：从 ModelScope 下载 OC20 实例数据

```bash
modelscope download --dataset OneScience/oc20 --local_dir ./data
```

方式二：使用集群共享数据（如已存在）

```bash
mkdir -p data/oc20
cp -r /path/to/s2ef_200k_uncompressed data/oc20/
cp -r /path/to/s2ef_val_id_uncompressed data/oc20/
```

**数据格式转换**

下载的原始数据通常是 `.extxyz` 文件，需要先用 `scripts/create_uma_finetune_dataset.py` 转换为 ASE-lmdb 格式，并计算 `elem_refs` 和 `normalizer_rmsd`。该脚本支持以下任务：

| 任务 | 说明 |
| --- | --- |
| `oc20` | 催化（示例） |
| `oc22` | 氧化物催化（仅限 1P2） |
| `oc25` | （电）催化（仅 1P2） |
| `omat` | 无机材料 |
| `omol` | 分子 + 聚合物 |
| `odac` | MOFs |
| `omc` | 分子晶体 |

以 OC20 为例：

```bash
python scripts/create_uma_finetune_dataset.py \
    --train-dir data/oc20/s2ef_200k_uncompressed \
    --val-dir data/oc20/s2ef_val_id_uncompressed \
    --uma-task oc20 \
    --regression-tasks ef \
    --output-dir data/oc20_finetune \
    --num-workers 8
```

转换后生成：

```text
data/oc20_finetune/
├── train/                 # ASE-lmdb 训练数据
├── val/                   # ASE-lmdb 验证数据
└── data/                  # 生成的数据配置 yaml
    └── uma_conserving_data_task_energy_force.yaml
```

然后用 `scripts/update_demo_config.py` 把生成的 `elem_refs`、`normalizer_rmsd` 和数据路径更新到 demo 配置文件：

```bash
python scripts/update_demo_config.py --demo-config demo/configs/oc20_ef_4dcu.yaml
```

`demo/run.sh` 会自动将仓库根目录作为 `ONESCIENCE_DATASETS_DIR`，因此配置文件中的相对路径会自动匹配。

如需真实训练或推理，还需准备 UMA checkpoint 和旋转基文件 `Jd.pt`，并放到：

```text
weight/uma-s-1p1_converted.pt
weight/Jd.pt
```

`demo/run.sh` 和 `inference/` 下的示例脚本都会自动检测 `weight/Jd.pt` 并设置 `ONESCIENCE_UMA_JD_PATH`。

**预检（不启动训练）**

```bash
bash demo/run.sh --config demo/configs/oc20_ef_4dcu.yaml --dry-run
```

**运行样例训练**

```bash
bash demo/run.sh --config demo/configs/oc20_ef_4dcu.yaml
```

SLURM 提交：

```bash
bash demo/run.sh --config demo/configs/oc20_ef_4dcu.yaml --submit
```

训练完成后，输出目录中会生成实验子目录：

```text
demo/outputs/
├── oc20_ef_4dcu_YYYYmmdd_HHMMSS/
│   ├── config.yaml
│   ├── hydra_config.yaml
│   ├── train_merged.out
│   └── uma_finetune_runs/
```

### 4. 常用训练参数

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `--config` | `run.sh` 使用的 YAML 配置文件 | `demo/configs/oc20_ef_4dcu.yaml` |
| `--dry-run` | 仅生成训练命令和 Hydra 配置预览 | 调试用 |
| `--submit` | 生成并提交 SLURM 作业 | 集群训练使用 |
| `launch.num_gpus` | 单节点使用的 GPU/DCU 数量 | `4` |
| `data.dataset_name` | UMA 任务数据集名 | `oc20`、`oc22`、`oc25`、`omat`、`omol`、`odac`、`omc` |
| `data.train_dataset.splits.train.src` | 训练集 ASE-lmdb 目录 | `data/oc20_finetune/train` |
| `data.val_dataset.splits.val.src` | 验证集 ASE-lmdb 目录 | `data/oc20_finetune/val` |
| `runner.train_eval_unit.model.checkpoint_location` | 微调 checkpoint 路径 | `weight/uma-s-1p1_converted.pt` |
| `epochs` | 训练轮数 | `1` |
| `batch_size` | 每卡 batch 大小 | `2` |
| `evaluate_every_n_steps` | 验证间隔步数 | `100` |

---

## 数据格式

UMA 微调支持以 `.extxyz` 作为多种任务的原始输入，但需要先用 `scripts/create_uma_finetune_dataset.py` 转换为 ASE-lmdb 格式。支持的 `--uma-task` 包括 `oc20`、`oc22`、`oc25`、`omat`、`omol`、`odac`、`omc`。

以 OC20 为例：

```bash
python scripts/create_uma_finetune_dataset.py \
    --train-dir data/oc20/s2ef_200k_uncompressed \
    --val-dir data/oc20/s2ef_val_id_uncompressed \
    --uma-task oc20 \
    --regression-tasks ef \
    --output-dir data/oc20_finetune \
    --num-workers 8
```

转换后的目录结构如下：

```text
data/oc20_finetune/
├── train/
│   ├── data.0000.aselmdb
│   ├── ...
├── val/
│   ├── data.0000.aselmdb
│   ├── ...
└── data/
    └── uma_conserving_data_task_energy_force.yaml
```

每个 lmdb 目录包含 ASE 原子的序列化数据。`data.elem_refs` 和 `data.normalizer_rmsd` 需与数据生成脚本输出一致，可通过 `scripts/update_demo_config.py` 自动同步到 demo 配置文件。

---

## OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

---

## 引用与许可证

- UMA 示例代码来自 OneScience 项目中的 matchem 示例实现，并参考了上游 fairchem 项目（https://github.com/facebookresearch/fairchem）。上游 fairchem 仓库软件以 [MIT License](https://fair-chem.github.io/core/install.html#license) 发布；fairchem 各模型 checkpoint 和数据集可能带有各自独立的许可证，使用时请遵循对应说明。
- 如果在科研工作中使用 UMA 微调或推理结果，建议引用 UMA/相关通用材料相互作用模型方法、fairchem/OneScience 相关项目信息和实际使用的数据集来源。

