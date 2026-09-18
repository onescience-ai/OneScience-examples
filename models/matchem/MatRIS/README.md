<p align="center">
  <strong>
    <span style="font-size: 30px;">MatRIS</span>
  </strong>
</p>


# 模型介绍

MatRIS 是面向材料表征与相互作用模拟的基础模型，全称为 Materials Representation and Interaction Simulation，可用于晶体结构的能量、力、应力和磁矩预测，并支持基于 ASE 和 pymatgen 结构对象的结构弛豫。


# 模型描述

MatRIS 基于图神经网络架构，使用 OMat24 和 MPTrj 等材料数据集进行训练，面向晶体材料开展能量、力、应力和磁矩预测及结构优化。

# 适用场景

| 场景 | 说明 |
| :---: | :---: |
| 晶体能量预测 | 输入 CIF、pymatgen Structure 或 ASE Atoms 结构，预测体系能量 |
| 力和应力预测 | 为结构弛豫、分子动力学或后续模拟提供力和应力估计 |
| 磁矩预测 | 在 `efsm` 任务下输出结构相关磁矩结果 |
| 结构弛豫前处理 | 使用 `StructOptimizer` 对候选晶体结构进行原子位置和晶胞优化 |
| 环境连通性验证 | 使用 `cif_file/demo.cif` 和轻量 MatRIS 模型检查 OneScience matchem 环境是否可用 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于模块验证和小规模前向验证，但结构弛豫速度较慢，不建议用于正式批量推理。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/MatRIS --local_dir ./matris
cd matris
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

### 训练权重

推理测试需要预训练模型。MatRIS 提供以下模型 key：

| 模型 key | 说明 |
| --- | --- |
| `matris_10m_omat` | 在 OMat24 数据集上训练 |
| `matris_10m_oam` | 在 OMat24 上训练，并在 sAlex+MPtrj 上微调 |
| `matris_10m_mp` | 在 MPTrj 数据集上训练 |

本仓库 `weight/` 目录已包含预训练权重。运行 `scripts/test_relaxation.py` 时会从 `weight/` 加载。


### 推理

**运行模块化验证**

```bash
python scripts/test_modularization.py
```

该脚本会实例化一个轻量 MatRIS 模型（随机初始化权重），并完成一次 CPU 前向传播，验证模型模块的连通性。

**运行结构弛豫推理**

```bash
python scripts/test_relaxation.py
```

该脚本读取 `cif_file/demo.cif`，通过 `StructOptimizer` 执行结构弛豫。

推理完成后，日志中会输出结构弛豫过程，并在内存中得到能量、力、应力、磁矩和最终结构对象。

**其他推理实例**

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

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- MatRIS 上游材料使用 BSD-3-Clause License。本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

- 如果在科研工作中使用 MatRIS 结果，建议引用 MatRIS 原始项目、OneScience 相关项目信息，并根据实际任务补充材料数据集、结构优化工具或下游分析工具引用。



