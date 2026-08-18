<p align="center">
  <strong>
    <span style="font-size: 30px;">NequIP</span>
  </strong>
</p>

# 模型介绍

NequIP 是面向分子和材料体系的机器学习原子间势（MLIP）模型，基于 E(3)-等变图神经网络构建，可对原子结构进行能量和受力预测。

参考实现：https://github.com/mir-group/nequip

# 模型描述

本目录提供 OneScience 集成的 NequIP 示例代码、模型包源码镜像以及可直接运行的训练、微调和推理示例。`model/` 目录仅对应 OneScience 主仓库中的 `src/onescience/models/nequip/`；训练工具、数据处理工具和其他共享模块由已安装的 OneScience 软件包提供。

本 examples 仓库不包含模型权重、训练数据、输出、缓存或 Slurm 日志。模型权重和数据集分别通过 ModelScope 下载。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 原子间势训练 | 使用示例配置和 ASE extxyz 数据训练 NequIP 模型 |
| 预训练模型微调 | 使用 OAM-L package 和带有 energy/forces 标注的数据进行微调 |
| 单点能量与受力推理 | 使用编译模型或微调 checkpoint 预测结构的能量、原子力和应力 |
| 结构弛豫 | 使用 ASE 对原子位置进行优化 |
| 能量-体积曲线 | 扫描周期晶体体积并计算对应能量 |
| Slurm/DCU 训练 | 使用仓库提供的配置和启动脚本提交单卡或多卡作业 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行训练。
- CPU 可以用于导入和小配置连通性验证，完整训练速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/nequip --local_dir ./nequip
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本目录不内置训练数据。以 FCC Cu 入门数据集为例，从 ModelScope 下载并放到当前示例目录的 `data/` 下：

```bash
modelscope download --dataset OneScience/FCC_Cu --local_dir ./data
```

下载后原始数据路径为 `data/data/FCC_Cu/raw/fcu.xyz`。该数据集包含 6,855 个结构，每个结构包含 52 个原子，元素为 C、H、O 和 Cu，文件为 ASE 可读取的 extxyz 格式，并包含周期晶格、energy 和 forces 标注。正式训练或微调时，请使用与目标体系、标签定义和单位一致的数据。

训练脚本通过共享目录读取模型和数据。运行训练或微调前，请根据实际集群路径设置：

```bash
export ONESCIENCE_MODELS_DIR=/path/to/onescience-models
export ONESCIENCE_DATASETS_DIR=/path/to/onescience-datasets
mkdir -p "$ONESCIENCE_MODELS_DIR/NequIP"
cp nequip/weight/NequIP-OAM-L-0.1.nequip.pth "$ONESCIENCE_MODELS_DIR/NequIP/"
cp nequip/weight/NequIP-OAM-L-0.1.nequip.zip "$ONESCIENCE_MODELS_DIR/NequIP/"
```

FCC Cu 数据集单独发布于 [OneScience/FCC_Cu](https://modelscope.cn/datasets/OneScience/FCC_Cu)。

### 训练

生成最小 smoke 数据并运行本地训练：

```bash
python demo/prepare_smoke_data.py
bash demo/run.sh --config configs/tutorial_smoke.yaml
```

下载官方 fcu 教程数据并提交训练：

```bash
python demo/download_tutorial_data.py
bash demo/run.sh --config configs/tutorial_fcu.yaml --submit
```

8 DCU 配置会根据当前资源自动本地运行或提交到 Slurm：

```bash
bash demo/run.sh --config configs/tutorial_smoke_8dcu.yaml
bash demo/run.sh --config configs/tutorial_fcu_8dcu.yaml
```

输出默认写入 `outputs/`，训练作业的实际等待时间取决于集群队列和可用资源。

### 训练权重

本目录不包含权重。请先下载 ModelScope 模型包，并使用其中的：

```text
nequip/weight/NequIP-OAM-L-0.1.nequip.pth
nequip/weight/NequIP-OAM-L-0.1.nequip.zip
```

### 微调

使用生成的 smoke 数据验证 OAM-L 微调流程：

```bash
python demo/prepare_smoke_data.py
bash demo/run.sh --config configs/oam_l_finetune_smoke.yaml --submit
```

使用正式微调配置：

```bash
bash demo/run.sh --config configs/oam_l_finetune.yaml --submit
```

正式微调数据应通过 `ONESCIENCE_DATASETS_DIR` 提供，并使用 ASE 可读取的 extxyz 格式；每帧至少应包含 `energy` 和 `forces`，元素类型、单位和标签定义必须与 OAM-L package 及配置保持一致。

### 推理

使用从 ModelScope 下载的编译模型进行单点能量、原子力和应力预测：

```bash
python single_point.py --compiled-model nequip/weight/NequIP-OAM-L-0.1.nequip.pth
python single_point.py \
  --compiled-model nequip/weight/NequIP-OAM-L-0.1.nequip.pth \
  --input structure.cif \
  --output outputs/single_point.json
```

计算能量-体积曲线并执行结构弛豫：

```bash
python energy_volume.py
python structure_relaxation.py --fmax 0.05 --steps 100 --output-dir outputs/oam_l_relax
```

使用微调生成的 checkpoint 推理：

```bash
python single_point.py \
  --checkpoint outputs/<run>/checkpoints/best.ckpt \
  --package nequip/weight/NequIP-OAM-L-0.1.nequip.zip \
  --output outputs/<run>/single_point.json
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- NequIP 相关代码来自 OneScience 项目中的 MatChem 集成，并参考了上游 NequIP 项目（https://github.com/mir-group/nequip）。OneScience 集成代码遵循主仓库中的 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)。
- 如果在科研工作中使用 NequIP 或 OAM-L 训练结果，建议引用 NequIP 原始论文、OneScience 相关项目信息和实际使用的数据集来源。
- OAM-L 模型权重不随本 examples 仓库发布，其再分发权限应以 OneScience/OAM-L 的原始发布条款为准。
- FCC Cu 数据集单独发布于 [OneScience/FCC_Cu](https://modelscope.cn/datasets/OneScience/FCC_Cu)，其许可和来源信息以数据集卡片及上游来源说明为准。
