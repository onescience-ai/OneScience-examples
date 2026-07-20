# MatterSim

MatterSim 是微软研究院提出的跨元素、温度和压力的深度学习原子间势模型，可用于无机材料、分子及周期性体系的能量、受力预测，并支持结构弛豫、分子动力学和自定义数据集微调。

论文：*MatterSim: A deep-learning atomistic model across elements, temperatures, and pressures*  
参考实现：[MatterSim 官方 GitHub](https://github.com/microsoft/mattersim)

本目录是 OneScience Examples 的 `models/matchem/MatterSim` 示例，面向 OneCode 自动化运行和本地快速验证场景。

---

## 仓库说明

当前支持能力：

- 单点能量与受力预测
- 多结构批量推理
- 晶体结构弛豫（BFGS / FIRE + ExpCellFilter / FrechetCellFilter）
- 分子动力学（NVT Berendsen / NVT Nose-Hoover）
- 自定义数据集微调训练（支持 DDP 多卡）
- GPU/DCU 优先运行

当前限制：

- 本目录不内置预训练权重和训练数据，需从 ModelScope 模型仓库 `OneScience/Mattersim` 下载
- 不提供独立在线推理服务、部署脚本或可视化页面

---

## 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 单点能量/受力预测 | 对给定原子结构快速预测能量和原子受力 |
| 批量结构推理 | 对多个结构进行批量能量/受力预测 |
| 结构弛豫 | 使用 FIRE/BFGS 优化原子位置和晶胞形状 |
| 分子动力学 | NVT 系综下运行短程 MD 采样 |
| 自定义数据微调 | 在自有数据集上微调预训练 MatterSim 模型 |
| 环境连通性验证 | 使用单点/弛豫脚本检查 OneScience matchem 环境、模型加载和 CUDA/DCU 可用性 |

---

## 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` | 工程使用说明文档 | 本文件 |
| `scripts/` | 推理与微调示例脚本 | 包含 `single_point.py`、`batch_inference.py`、`relax.py`、`md.py`、`finetune.py` 和 `finetune_config.yaml` |

> 模型源码和预训练权重请从 ModelScope 模型仓库 `OneScience/Mattersim` 下载。

---

## 使用说明

### 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

### 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
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

**下载模型包**

本示例不内置权重和数据，请从 ModelScope 下载完整模型包：

```bash
modelscope download --model OneScience/Mattersim --local_dir ./Mattersim
```

下载后目录结构如下：

```text
Mattersim/
├── model/                      # MatterSim 模型源码
├── weight/
│   └── mattersim-v1.0.0-1M.pth # 预训练 checkpoint
├── data/
│   └── high_level_water.xyz    # 示例微调数据
└── ...
```

**进入示例目录**

本示例位于 OneScience Examples 的 `models/matchem/MatterSim`，进入该目录后所有命令均相对于该目录执行：

```bash
cd models/matchem/MatterSim
```

**单点推理**

```bash
cd scripts
python single_point.py --checkpoint ../Mattersim/weight/mattersim-v1.0.0-1M.pth
```

**批量推理**

```bash
cd scripts
python batch_inference.py --checkpoint ../Mattersim/weight/mattersim-v1.0.0-1M.pth
```

**结构弛豫**

```bash
cd scripts
python relax.py --device cuda
```

> 默认使用 `../Mattersim/weight/mattersim-v1.0.0-1M.pth`。

**分子动力学**

```bash
cd scripts
python md.py --device cuda
```

> 同样默认使用 `../Mattersim/weight/mattersim-v1.0.0-1M.pth`。

**微调**

直接修改 `scripts/finetune_config.yaml` 中的路径和参数（例如 `train_data_path`、`checkpoint` 等）：

```bash
cd scripts
# 编辑 finetune_config.yaml 中的 train_data_path、checkpoint 等字段
```

运行：

```bash
python finetune.py --config finetune_config.yaml
```

多卡 DDP：

```bash
torchrun --nproc_per_node=4 finetune.py --config finetune_config.yaml
```

微调完成后，输出目录 `results/mattersim/` 中会生成：

```text
results/mattersim/
├── best_model.pth
├── last_model.pth
└── training logs
```

---

## 常用参数

`scripts/finetune_config.yaml` 中主要字段说明：

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `train_data_path` | 训练数据文件路径 | `../Mattersim/data/high_level_water.xyz` |
| `valid_data_path` | 验证数据文件路径，不需要可设为 `null` | `null` |
| `checkpoint` | 预训练 checkpoint 路径 | `../Mattersim/weight/mattersim-v1.0.0-1M.pth` |
| `save_path` | 微调结果输出目录 | `./results/mattersim` |
| `epochs` | 训练轮数 | `1` |
| `batch_size` | 训练 batch 大小 | `16` |
| `lr` | 学习率 | `2.0e-4` |
| `device` | 计算设备 | `cuda` / `cpu` |
| `cutoff` | 近邻截断半径 | `5.0` |
| `threebody_cutoff` | 三体截断半径 | `4.0` |
| `include_forces` | 是否训练力 | `true` |
| `include_stresses` | 是否训练应力 | `false` |
| `save_checkpoint` | 是否保存 checkpoint | `true` |

命令行参数可以覆盖 YAML 中的值，例如临时改为 2 个 epoch：

```bash
python finetune.py --config finetune_config.yaml --epochs 2
```

---

## 数据格式

MatterSim 推理与微调支持 ASE 可读的常见原子结构格式，例如：

- `.xyz` / `.extxyz`
- `.cif`
- `.vasp` / POSCAR

训练/验证文件需要包含能量和力字段，字段名默认为 `energy` 和 `forces`。如果字段名不同，可通过命令行参数或修改 `finetune_config.yaml` 覆盖。

---

## OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

---

## 引用与许可证

- MatterSim 相关代码来自 OneScience 项目中的 matchem 示例实现，并参考了上游 MatterSim 项目（https://github.com/microsoft/mattersim）。上游 MatterSim 代码以 [MIT License](https://github.com/microsoft/mattersim/blob/main/LICENSE) 发布。
- 如果在科研工作中使用 MatterSim 训练或推理结果，建议引用 MatterSim 原始论文、OneScience 相关项目信息和实际使用的数据集来源。
