---
license: mit
language:
- en
- zh
tags:
- OneScience
- MACE
- 机器学习力场
- 分子模拟
- 材料计算
- 图神经网络
- 等变神经网络
- 训练
- 推理
frameworks: PyTorch
---

# MACE

MACE 是面向分子和材料体系的机器学习原子间势（MLIP）模型，基于 **E(3)-等变图神经网络（E(3)-Equivariant GNN）** 构建，可从原子结构数据中学习体系能量、原子受力，并支持基于 HDF5/XYZ 数据的训练与验证流程。

论文：*MACE: Higher order equivariant message passing neural networks for fast and accurate force fields*  
参考实现：[MACE 官方 GitHub](https://github.com/ACEsuit/mace)

---

## 仓库说明

本仓库是 OneScience 整理的 MACE 最小可运行独立模型仓库，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 训练：使用 HDF5/XYZ 数据训练 MACE 原子间势。
- 评测：训练过程中按 `eval_interval` 在验证集上输出能量/力误差指标。
- 预检：检查配置、数据文件、statistics 和关键脚本是否齐全。
- 分布式训练：支持单卡、单节点多卡（torchrun）以及 SLURM 多节点提交。

当前不支持能力：

- 不内置预训练权重。
- 不内置真实训练数据集下载与预处理。
- 不提供独立推理服务、部署脚本或可视化页面。

---

## 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 原子间势训练 | 使用标准配置读取 HDF5/XYZ 数据并训练 MACE 模型 |
| 分布式训练预检 | 检查多卡/多节点训练配置、数据路径和 statistics 是否一致 |
| 验证集评测 | 训练过程中在验证集上输出能量和力相关误差指标 |
| 自有数据迁移 | 参考现有配置替换为自有 HDF5/XYZ 数据和 statistics 文件 |
| 环境连通性验证 | 使用预检脚本检查 OneScience matchem 环境、pyyaml、h5py 和数据可读性 |

---

## 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 本文件 |
| `model/` | MACE 模型源码 | 包含 `__init__.py`、`mace.py`、`__version__.py`、`py.typed` |
| `scripts/train.py` | MACE 训练主入口 | 来自 OneScience matchem  |
| `scripts/demo/run.sh` | 统一训练入口 | 支持直接运行、dry-run 和 SLURM 提交 |
| `scripts/demo/_parse_config.py` | 配置解析脚本 | 生成训练命令、环境变量和预检文件列表 |
| `scripts/demo/configs/` | 训练配置文件 | 包含 `DMC.yaml`、`ani1x_8dcu.yaml` 等 |
| `scripts/demo/templates/` | 脚本模板 | 环境初始化、预检、SLURM header 模板 |
| `scripts/fasteq/` | Fast Equilibration 辅助脚本 | 可选的加速平衡工具 |

---

## 使用说明

### 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

### 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行训练。
- CPU 可以用于导入、配置检查和小数据连通性验证，完整训练速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 26.04 或与当前集群匹配的 OneScience 推荐版本。

**软件要求**

- Python 3.11
- torch、e3nn、torch_ema
- pyyaml、h5py
- OneScience matchem 运行环境

OneScience 安装方式（参考 [OneScience README](https://gitee.com/onescience-ai/onescience)）：

```bash
git clone https://gitee.com/onescience-ai/onescience.git
cd onescience
bash install.sh matchem
```

### 3. 快速开始

**进入示例目录**

本示例位于 OneScience Examples 的 `models/matchem/MACE`，进入该目录后所有命令均相对于该目录执行：

```bash
cd models/matchem/MACE
```

**准备数据**

本仓库不内置训练数据。请准备 HDF5/XYZ 格式数据并放到仓库根目录的 `data/` 下。以 DMC 数据集为例，预期目录结构如下：

```text
data/
└── data/
    └── DMC/
        ├── solvent_xtb_train_200.xyz
        └── solvent_xtb_test.xyz
```

`scripts/demo/run.sh` 会自动将仓库根目录作为 `ONESCIENCE_DATASETS_DIR`，因此无需手动设置该变量即可匹配配置文件中的路径。

其他配置（如 `ani1x_8dcu.yaml`、`water_*.yaml` 等）需要下载对应数据集并调整 YAML 中的数据路径。

**预检（不启动训练）**

```bash
bash scripts/demo/run.sh --config scripts/demo/configs/DMC.yaml --dry-run
```

**运行样例训练**

单卡/直接运行：

```bash
bash scripts/demo/run.sh --config scripts/demo/configs/DMC.yaml
```

多卡（torchrun）：

```bash
# 以 8 卡为例，配置文件中 launch.launcher 应为 torchrun
bash scripts/demo/run.sh --config scripts/demo/configs/ani1x_8dcu.yaml
```

SLURM 提交：

```bash
bash scripts/demo/run.sh --config scripts/demo/configs/ani1x_8dcu.yaml --submit
```

训练完成后，输出目录中会生成实验子目录，通常包含配置快照、训练日志、checkpoint 和验证指标：

```text
scripts/demo/outputs/
├── DMC_YYYYmmdd_HHMMSS/
│   ├── config.yaml
│   ├── training logs
│   ├── checkpoints
│   └── final_model
```


### 4. 常用训练参数

`scripts/demo/configs/xxx.yaml` 中主要字段说明：

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `name` | 实验名称 | `DMC` |
| `train_args.model` | 模型类型 | `MACE` |
| `train_args.train_file` | 训练 HDF5 目录或 XYZ 文件 | `data/ani1x/ANI1x_cc_DFT_rc5_train` |
| `train_args.valid_file` | 验证 HDF5 目录或 XYZ 文件 | `data/ani1x/ANI1x_cc_DFT_rc5_val` |
| `train_args.test_file` | 测试 HDF5 目录或 XYZ 文件 | `data/ani1x/ANI1x_cc_DFT_rc5_test` |
| `train_args.statistics_file` | statistics 文件 | `data/ani1x/ANI1x_cc_DFT_rc5_statistics.json` |
| `train_args.r_max` | 截断半径，需与 statistics 一致 | `5.0` |
| `train_args.batch_size` | 训练 batch 大小 | `128` |
| `train_args.max_num_epochs` | 最大训练轮数 | `20` |
| `train_args.swa` | 启用随机权重平均 | `true` |
| `launch.num_gpus` | 单节点使用的 GPU/DCU 数量 | `8` |
| `launch.launcher` | 启动方式 | `python` / `torchrun` |

---

## 数据格式

MACE 训练数据支持两种主要格式：

### 1. HDF5 分片目录

将 HDF5 文件组织为目录，每个文件包含标准字段（由 OneScience datapipes.materials 定义），例如：

- 原子位置、能量、受力、原子类型、边索引等。
- 通常与 `statistics_file` 配套使用。

### 2. extended XYZ 文件

标准的 ASE extxyz 文件，需包含能量和力字段，字段名在 YAML 中通过 `energy_key` 和 `forces_key` 指定：

```yaml
train_args:
  train_file: "${ONESCIENCE_DATASETS_DIR}/data/data/DMC/solvent_xtb_train_200.xyz"
  energy_key: energy_xtb
  forces_key: forces_xtb
```

### 3. statistics 文件

JSON 文件，包含元素参考能量、平均邻居数、`r_max` 等统计信息。配置中的 `r_max` 必须与 statistics 文件中的 `r_max` 一致。

---

## OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

---

## 引用与许可证

- MACE 相关代码来自 OneScience 项目中的 matchem 示例实现，并参考了上游 MACE 项目（https://github.com/ACEsuit/mace）。上游 MACE 代码以 [MIT License](https://github.com/ACEsuit/mace/blob/main/LICENSE) 发布。
- 如果在科研工作中使用 MACE 训练结果，建议引用 MACE 原始论文、OneScience 相关项目信息和实际使用的数据集来源。
