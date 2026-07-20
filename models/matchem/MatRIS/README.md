<p align="center">
  <strong>
    <span style="font-size: 30px;">MatRIS</span>
  </strong>
</p>


# 模型介绍

MatRIS 是面向材料表征与相互作用模拟的基础模型，全称为 Materials Representation and Interaction Simulation，可用于晶体结构的能量、力、应力和磁矩预测，并支持基于 ASE 和 pymatgen 结构对象的结构弛豫。它适合材料结构性质快速评估、候选晶体结构初筛、结构优化前处理、势能面近似验证和材料模拟流程连通性检查等场景。


# 仓库说明

本仓库是 OneScience 整理的 MatRIS 最小可运行模型仓库，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- CPU 前向验证：使用极简晶体结构检查 MatRIS 模型能否完成能量、力、应力和磁矩前向传播。
- 结构弛豫示例：读取 `cif_file/demo.cif`，并通过 `StructOptimizer` 执行结构优化流程。
- GPU/DCU 优先运行：支持在已配置 OneScience matchem 环境的计算卡节点上运行。

当前不支持能力：

- 不支持训练、微调和完整材料性质评测流程。
- 不内置结构可视化服务。
- 不内置全部预训练 checkpoint，运行结构弛豫时需当前环境可解析或下载 MatRIS 模型 key。

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 晶体能量预测 | 输入 CIF、pymatgen Structure 或 ASE Atoms 结构，预测体系能量 |
| 力和应力预测 | 为结构弛豫、分子动力学或后续模拟提供力和应力估计 |
| 磁矩预测 | 在 `efsm` 任务下输出结构相关磁矩结果 |
| 结构弛豫前处理 | 使用 `StructOptimizer` 对候选晶体结构进行原子位置和晶胞优化 |
| 环境连通性验证 | 使用 `cif_file/demo.cif` 和轻量 MatRIS 模型检查 OneScience matchem 环境是否可用 |

# 文件说明

| 路径 | 功能 | 备注 |
| :---: | :---: | :---: |
| `README.md` |  工程使用说明文档 | 中文为主 |
| `model/__init__.py` | MatRIS 模型包入口 | 暴露 `MatRIS`、`GraphConverter`、`RadiusGraph` |
| `model/matris.py` | MatRIS 模型定义 | MatRIS 核心模型实现 |
| `download.sh` | 大文件下载脚本 | 从 ModelScope 下载 `weight/` 和 `cif_file/` |
| `scripts/test_modularization.py` | MatRIS 模块化验证脚本  | 检查导入、实例化、CPU 和 CUDA 前向；使用随机权重 |
| `scripts/test_relaxation.py` | 结构弛豫示例入口  | 使用 `matris_10m_oam` 预训练模型 key；读取 `cif_file/demo.cif` |
| `cif_file/demo.cif` |  最小结构弛豫示例输入 | 由 `download.sh` 从 ModelScope 下载 |
| `weight/` | 预训练权重目录 | 由 `download.sh` 从 ModelScope 下载官方权重 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于模块验证和小规模前向验证，但结构弛豫速度较慢，不建议用于正式批量推理。
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

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光 DCU：

```bash
hy-smi
```

## 3. 预训练模型与示例结构

推理测试需要预训练模型和示例 CIF 文件。由于文件较大，本仓库不直接包含 `weight/` 和 `cif_file/`，可通过 `download.sh` 从 ModelScope 完整模型包下载：

```bash
bash download.sh
```

下载完成后，目录结构如下：

```text
weight/
├── MatRIS_10M_OAM.pth.tar
└── MatRIS_10M_MP.pth.tar
cif_file/
└── demo.cif
```

MatRIS 提供以下模型 key：

| 模型 key | 说明 |
| --- | --- |
| `matris_10m_omat` | 在 OMat24 数据集上训练 |
| `matris_10m_oam` | 在 OMat24 上训练，并在 sAlex+MPtrj 上微调 |
| `matris_10m_mp` | 在 MPTrj 数据集上训练 |

运行 `scripts/test_relaxation.py` 时会从 `weight/` 加载；若本地不存在对应权重，也会尝试从 figshare 自动下载到 `weight/` 目录下。

**注意：推理测试不需要额外材料数据集。** 只有训练、微调或完整评测才需要。

## 4. 快速开始

### 进入示例目录

本示例位于 OneScience Examples 的 `models/matchem/MatRIS`，进入该目录后所有命令均相对于该目录执行：

```bash
cd models/matchem/MatRIS
```

### 下载大文件

运行推理脚本前，先从 ModelScope 下载预训练权重和示例 CIF：

```bash
bash download.sh
```

下载完成后会生成 `weight/` 和 `cif_file/` 目录。

### 运行模块化验证

```bash
python scripts/test_modularization.py
```

该脚本会实例化一个轻量 MatRIS 模型（随机初始化权重），并完成一次 CPU 前向传播，验证模型模块的连通性。

### 运行结构弛豫推理

```bash
python scripts/test_relaxation.py
```

该脚本读取 `cif_file/demo.cif`，通过 `StructOptimizer` 执行结构弛豫。

推理完成后，日志中会输出结构弛豫过程，并在内存中得到能量、力、应力、磁矩和最终结构对象：

```text
relaxation result
├── trajectory.energies[-1]
├── trajectory.forces[-1]
├── trajectory.stresses[-1]
├── trajectory.magmoms[-1]
└── final_structure
```

默认使用 `matris_10m_oam` 预训练模型 key，并根据当前环境自动选择 CUDA 或 CPU。若当前环境没有可用 GPU/DCU，脚本仍可在 CPU 上运行，但耗时会更长。

## 5. 推理示例

除结构弛豫外，也可以使用 `MatRISCalculator` 对单个结构进行能量、力、应力和磁矩预测：

```python
import torch
from ase.build import bulk
from onescience.utils.matris import MatRISCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
calc = MatRISCalculator(
    model="matris_10m_oam",
    task="efsm",
    device=device,
)

atoms = bulk("Cu", a=5.43, cubic=True)
atoms.calc = calc

energy = atoms.get_potential_energy()   # 总能量 (eV)
forces = atoms.get_forces()             # 力 (eV/Å)
stress = atoms.get_stress()             # 应力 (eV/Å³)
magmoms = atoms.get_magnetic_moments()  # 磁矩 (μB)
```

## 常用推理参数

可直接修改 `scripts/test_relaxation.py` 中的变量：

| 变量 | 说明 | 示例 |
| --- | --- | --- |
| `model_name` | MatRIS 预训练模型 key | `matris_10m_oam`、`matris_10m_mp` |
| `task` | 预测任务类型，控制输出能量、力、应力和磁矩 | `e`、`ef`、`efs`、`efsm` |
| `device` | 推理设备 | `cuda` 或 `cpu` |
| `max_steps` | 最大结构弛豫步数 | `500` |
| `fmax` | 力收敛阈值 | `0.05` |


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- MatRIS 上游材料使用 BSD-3-Clause License。本仓库保留来源说明，并面向 OneScience 社区使用场景进行整理。

- 如果在科研工作中使用 MatRIS 结果，建议引用 MatRIS 原始项目、OneScience 相关项目信息，并根据实际任务补充材料数据集、结构优化工具或下游分析工具引用。
