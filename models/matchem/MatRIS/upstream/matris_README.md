# MatRIS 推理 Demo

本目录提供 MatRIS 在 OneScience 中的基础验证、结构弛豫和 API 使用示例。MatRIS 当前主要用于材料结构的能量、力、应力、磁矩预测，以及基于 ASE/pymatgen 的结构优化和分子动力学。

## 快速上手

### 1. 准备环境

首次使用前安装 MatChem 基础环境：

```bash
cd /path/to/onescience/examples/matchem
bash matchem_install.sh
```

后续每次使用：

```bash
cd /path/to/onescience/examples/matchem
source matchem_env.sh
```

MatRIS 会优先从以下目录读取预训练模型：

```text
$ONESCIENCE_MODELS_DIR/matris/
```

当前代码支持的主要 checkpoint 文件名：

| 模型名 | checkpoint 文件 |
| --- | --- |
| `matris_10m_oam` | `MatRIS_10M_OAM.pth.tar` |
| `matris_10m_mp` | `MatRIS_10M_MP.pth.tar` |

如果 `$ONESCIENCE_MODELS_DIR/matris/` 下没有模型文件，代码会尝试下载到 `~/.cache/matris`。生产环境建议提前把 checkpoint 放到共享模型目录，避免运行时访问外网。

### 2. 运行模块验证

```bash
cd /path/to/onescience/examples/matchem/matris
python test_modularization.py
```

该脚本会检查：

- MatRIS 相关模块是否可以正常 import
- MatRIS 小模型是否可以实例化
- CPU 前向传播是否正常
- 如果有可用 GPU/DCU，是否可以执行 CUDA 前向传播

### 3. 运行结构弛豫示例

```bash
cd /path/to/onescience/examples/matchem/matris
python test_relaxation.py
```

示例会读取：

```text
cif_file/demo.cif
```

并使用：

```text
model_name = "matris_10m_oam"
task = "efsm"
optimizer = "FIRE"
```

完成原子位置和晶胞弛豫后，脚本会在内存中得到：

- `trajectory.energies[-1]`：最终能量
- `trajectory.forces[-1]`：最终力
- `trajectory.stresses[-1]`：最终应力
- `trajectory.magmoms[-1]`：最终磁矩
- `final_structure`：最终 pymatgen 结构

如需保存最终结构，可在 `test_relaxation.py` 末尾添加：

```python
final_structure.to(filename="relaxed.cif")
```

## 常用 API

### ASE Calculator

```python
from ase.build import bulk
import torch

from onescience.utils.matris import MatRISCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
calc = MatRISCalculator(
    model="matris_10m_oam",
    task="efsm",
    device=device,
)

atoms = bulk("Cu", a=3.61, cubic=True)
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
magmoms = atoms.get_magnetic_moments()
```

### 结构优化

```python
import torch
from pymatgen.core.structure import Structure

from onescience.utils.matris import StructOptimizer

device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = StructOptimizer(
    model="matris_10m_oam",
    task="efsm",
    optimizer="FIRE",
    device=device,
)

structure = Structure.from_file("cif_file/demo.cif")
result = optimizer.relax(
    atoms=structure,
    verbose=True,
    steps=500,
    fmax=0.05,
    relax_cell=True,
    ase_filter="FrechetCellFilter",
)

final_structure = result["final_structure"]
```

### 分子动力学

```python
from ase.build import bulk
import torch

from onescience.utils.matris import MolecularDynamics

device = "cuda" if torch.cuda.is_available() else "cpu"
atoms = bulk("Cu", a=3.61, cubic=True)

md = MolecularDynamics(
    atoms=atoms,
    model="matris_10m_oam",
    ensemble="nvt",
    temperature=300,
    timestep=1.0,
    trajectory="md_out.traj",
    logfile="md_out.log",
    loginterval=100,
    task="efsm",
    device=device,
)
md.run(1000)
```

## 任务类型

| `task` | 输出 |
| --- | --- |
| `e` | energy |
| `ef` | energy + forces |
| `efs` | energy + forces + stress |
| `efsm` | energy + forces + stress + magnetic moments |

## 目录结构

```text
matris/
  README.md
  test_modularization.py
  test_relaxation.py
  cif_file/
    demo.cif
  data/
    README.md
    requirements.txt
    pyproject.toml
```

## 常见问题

1. 模型文件找不到：确认 `$ONESCIENCE_MODELS_DIR/matris/MatRIS_10M_OAM.pth.tar` 或 `MatRIS_10M_MP.pth.tar` 存在。
2. 运行时尝试联网下载：说明共享模型目录没有 checkpoint，生产环境建议提前上传模型文件。
3. CUDA/DCU 不可用：`test_modularization.py` 会自动跳过 GPU 测试；推理和弛豫仍可在 CPU 上运行，但速度较慢。
4. 客户自有结构：把 `cif_file/demo.cif` 替换为自己的 CIF，或在脚本中修改 `Structure.from_file(...)` 的路径。
