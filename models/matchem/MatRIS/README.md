# MatRIS

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 MatRIS 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/matris" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

MatRIS 是材料表征与相互作用模拟模型，可面向晶体结构进行能量、力、应力和磁矩相关预测，并可通过 `StructOptimizer` 执行结构弛豫示例。本标准模型包整理了 OneScience 中的 MatRIS 示例脚本、demo CIF、上游说明，以及用于最小验证的 phonon reference CSV 数据。

本包没有新增训练配置，因为当前 MatRIS 示例目录没有 YAML 训练入口。标准运行能力以预检、CPU 前向验证和结构弛豫示例为主；如果运行结构弛豫，环境需能解析或下载 MatRIS 预训练模型 key，例如 `matris_10m_oam`。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | matchem |
| 领域标签 | matchem, materials, foundation_model |
| 任务 | materials_representation_interaction_simulation |
| 任务标签 | energy_prediction, force_prediction, stress_prediction, magmom_prediction, relaxation |
| 主平台资源 | https://modelscope.cn/models/OneScience/matris |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `examples/matchem/matris` |
| 支持能力 | 推理 / 评测 / 预检 |
| 必需模型文件 | `scripts/preflight_matris.py`, `test_modularization.py`, `test_relaxation.py`, `cif_file/demo.cif` |
| 必需数据集 | `OneScience/pbe` |
| 最小验证 | `python scripts/preflight_matris.py --data-dir data --skip-model-forward` |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 说明文档 | 模型用途、文件、下载、运行和诊断说明 | 是 | 全部能力 | `session_workdir/README.md` | 本文件 |
| `manifest.yaml` | Manifest 文件 | 标准默认机器可读运行说明 | 是 | 全部能力 | `session_workdir/manifest.yaml` | 与 `onescience_run_manifest.yaml` 内容一致 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 本次任务要求的大模型运行 Manifest 文件名 | 是 | 全部能力 | `session_workdir/onescience_run_manifest.yaml` | 修改后需与 `manifest.yaml` 保持一致 |
| `scripts/preflight_matris.py` | 预检脚本 | 检查 CSV 数据、MatRIS 模块导入和 CPU 前向传播 | 是 | 预检、评测 | `session_workdir/scripts/preflight_matris.py` | 最小验证入口 |
| `test_modularization.py` | 测试脚本 | 原始 MatRIS 模块化验证脚本 | 否 | 预检 | `session_workdir/test_modularization.py` | 可人工运行 |
| `test_relaxation.py` | 示例脚本 | 使用 `StructOptimizer` 对 `demo.cif` 做结构弛豫 | 否 | 推理 | `session_workdir/test_relaxation.py` | 依赖预训练模型 key |
| `cif_file/demo.cif` | 示例输入 | 结构弛豫示例 CIF 文件 | 否 | 推理 | `session_workdir/cif_file/demo.cif` | 原始示例文件 |
| `data/pbe.csv`, `data/pbesol.csv`, `data/pbe/` | 数据文件 | PBE/PBESOL phonon reference CSV 与 PBE YAML.BZ2 数据目录 | 是 | 预检、评测 | `session_workdir/data/` | 来自数据集仓库 `OneScience/pbe` |
| `upstream/` | 上游材料 | 上游 README、当前 OneScience MatRIS 使用说明、LICENSE、requirements、pyproject | 否 | 说明 | `session_workdir/upstream/` | 便于追溯 |

## Manifest

本仓库提供 `manifest.yaml` 和 `onescience_run_manifest.yaml`，两者内容一致。修改文件、下载命令、运行命令或数据关系后必须同步更新两个 Manifest。

## 模型 vs 数据集关系

模型仓库目标 ID 是 `OneScience/matris`，数据集仓库目标 ID 是 `OneScience/pbe`。模型 Manifest 的 `relations.required_datasets` 指向 `OneScience/pbe` 数据集，并提供完整 `resource_ref`。

## 文件与下载

```bash
modelscope download --model OneScience/matris --local_dir session_workdir
modelscope download --dataset OneScience/pbe --local_dir session_workdir
```

如果网页端使用 `--cache_dir` 下载模型，运行前必须切换到实际下载后的模型包根目录，再执行预检或推理命令。

## 环境安装

```bash
bash install.sh matchem
```

## 运行流程

### 1. 环境预检

```bash
python scripts/preflight_matris.py --data-dir data --skip-model-forward
```

完整 OneScience matchem 环境中可进一步执行：

```bash
python scripts/preflight_matris.py --data-dir data
```

### 2. 下载

```bash
modelscope download --model OneScience/matris --local_dir session_workdir
modelscope download --dataset OneScience/pbe --local_dir session_workdir
```

### 3. 应用运行包和准备文件

```bash
cd session_workdir
```

### 4. 运行前预检

```bash
python scripts/preflight_matris.py --data-dir data --skip-model-forward
```

### 5. 运行

```bash
python test_relaxation.py
```

### 6. 验证输出

轻量预检成功时会输出 CSV 行数；完整预检还会输出模块导入和 CPU 前向传播通过。结构弛豫示例成功时会生成能量、力、应力、磁矩和最终结构对象。

## 输出说明

最小验证输出在标准输出中。结构弛豫示例的主要结果包括 `trajectory.energies`、`trajectory.forces`、`trajectory.stresses`、`trajectory.magmoms` 和 `final_structure`。

## 预检与诊断

- `ModuleNotFoundError`：缺少 OneScience matchem、torch、pymatgen 或 ase。
- `数据文件不存在`：未下载数据集或 `data/` 路径不正确。
- `CSV 表头不匹配`：数据文件不是当前 phonon reference CSV。
- 预训练模型加载失败：当前环境无法解析 `matris_10m_oam`，请检查模型缓存或下载设置。

## 限制与适用范围

本包当前不提供训练或微调入口；主要用于 MatRIS 模块预检、CPU 前向验证、结构弛豫示例和 phonon reference CSV 读取验证。

## 引用与许可证

MatRIS 上游材料声明 BSD-3-Clause 许可证，详见 `upstream/LICENSE`。OneScience 集成代码以 OneScience 仓库许可证为准。
