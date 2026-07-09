<p align="center">
  <strong>
    <span style="font-size: 30px;">RFdiffusion</span>
  </strong>
</p>

# 模型介绍

RFdiffusion 是一种基于扩散模型的蛋白质骨架生成和设计方法，可用于无条件骨架生成、motif scaffolding、PPI/binder 设计和对称寡聚体采样。当前 ModelScope 包面向下载即用、本地快速验证和 OneCode 自动化运行场景，代码、配置、示例输入和权重均已放在当前目录内。

# 仓库说明

本仓库是 OneScience 整理的 RFdiffusion 全量可运行模型仓库。`examples/` 和 `weight/` 已随模型包完整提供，运行 `env_install.sh` 只安装依赖并执行预检，不再从社区下载权重或示例输入。

当前支持能力：

- 无条件蛋白骨架设计。
- motif scaffolding，使用本地或用户提供的 PDB。
- PPI/binder 设计，按 RFdiffusion 原始配置参数指定 hotspot、contig 等输入。
- 对称寡聚体采样，使用 `config/inference/symmetry.yaml`。
- 包完整性预检 `scripts/preflight.py`，可检查示例输入、真实权重和本地 import。

当前不支持能力：

- 不提供独立训练、微调或评估脚本。
- 不支持运行时自动访问远端补齐缺失文件；请保持模型包下载完整。
- 不需要也不内置 Protenix/AlphaFold 类模型使用的 CCD 缓存或 MSA 数据库。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 无条件骨架生成 | 输入 contig 约束，输出设计骨架 PDB。 |
| Motif scaffolding | 输入包含 motif 的 PDB 和 contig 约束，输出 scaffold 设计结果。 |
| PPI/binder 设计 | 输入目标结构、hotspot 和 contig 参数，输出 binder 设计候选。 |
| 对称寡聚体采样 | 使用 symmetry 配置生成对称结构设计。 |
| ModelScope 全量包验证 | 使用包内 `config/ modules/ scripts/ examples/ weight/` 布局直接预检和推理。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | ModelScope 元信息 | RFdiffusion 推理/采样包 |
| `config/inference/` | Hydra 推理配置 | 包含 `base.yaml` 和 `symmetry.yaml` |
| `modules/` | RFdiffusion 和 SE3 transformer 源码依赖 | 已改为本地 `modules.*` 导入 |
| `scripts/run_inference.py` | 推荐推理入口 | 支持 Hydra 参数覆盖 |
| `scripts/run_inference.sh` | Shell 推理入口 | 默认调用 `scripts/run_inference.py` |
| `scripts/preflight.py` | 包完整性预检 | `--strict-weights` 可检查真实权重 |
| `examples/input_pdbs/` | 最小示例 PDB | 默认使用 `1qys.pdb`，motif 示例使用 `1YCR.pdb` |
| `weight/` | RFdiffusion checkpoint 目录 | 默认采样使用 `Base_ckpt.pt` |
| `MODEL_FILE_MANIFEST.tsv` | 模型文件清单 | 记录权重和示例输入文件信息 |

# 权重和示例输入

当前 ModelScope 包内已包含默认示例 PDB 和 RFdiffusion 权重，下载完整模型包后可以直接使用。

当前 `weight/` 中应包含以下真实权重，文件名需保持不变：

```text
weight/ActiveSite_ckpt.pt
weight/Base_ckpt.pt
weight/Base_epoch8_ckpt.pt
weight/Complex_Fold_base_ckpt.pt
weight/Complex_base_ckpt.pt
weight/Complex_beta_ckpt.pt
weight/InpaintSeq_Fold_ckpt.pt
weight/InpaintSeq_ckpt.pt
weight/RF_structure_prediction_weights.pt
```

默认示例输入包括：

```text
examples/input_pdbs/1qys.pdb
examples/input_pdbs/1YCR.pdb
```

RFdiffusion 首次推理可能会在 `.cache/schedules/` 生成 IGSO3 schedule 派生缓存，这是本地计算缓存，不是外部数据库。

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用GPU或DCU运行。
- CPU可以用于连通性验证，但速度较慢。
- DCU用户需要预先安装DTK，建议使用DTK 25.04.2以上版本或与当前集群匹配的OneScience推荐版本。

**软件要求**

想了解更多适配内容请联系 liubiao@sugon.com

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光DCU：

```bash
hy-smi
```

## 快速开始

### 1. 安装onescience库

```bash
git clone https://gitee.com/onescience-ai/onescience
cd onescience
bash install.sh bio
```

### 2. 下载模型包并下载权重&案例文件

```bash
modelscope download --model OneScience/RFdiffusion --local_dir ./RFdiffusion
cd ./RFdiffusion
bash download_assets.sh
```

### 3. 运行预检

检查文件和真实权重：

```bash
python scripts/preflight.py --strict-weights
```

安装依赖后检查本地 import：

```bash
python scripts/preflight.py --strict-weights --strict-imports
```

只验证入口和 Hydra 配置，不执行采样：

```bash
RF_DIFFUSION_SMOKE_TEST=1 python scripts/run_inference.py
```

### 4. 运行推理

无条件骨架采样示例：

```bash
python scripts/run_inference.py \
  'contigmap.contigs=[80-80]' \
  diffuser.T=15 \
  inference.final_step=15 \
  inference.num_designs=1 \
  inference.write_trajectory=False \
  inference.output_prefix=outputs/smoke/design
```

Motif scaffolding 示例：

```bash
python scripts/run_inference.py \
  inference.input_pdb=examples/input_pdbs/1YCR.pdb \
  'contigmap.contigs=[10-40/A163-181/10-40]' \
  inference.output_prefix=outputs/motif/design
```

对称采样示例：

```bash
python scripts/run_inference.py --config-name symmetry \
  diffuser.T=15 \
  inference.final_step=15 \
  inference.output_prefix=outputs/symmetry/c2
```

### 5. 常用环境变量

```bash
export RF_DIFFUSION_MODEL_DIR=weight
export RF_DIFFUSION_INPUT_PDB=examples/input_pdbs/1qys.pdb
export RF_DIFFUSION_OUTPUT_PREFIX=outputs/design
export RF_DIFFUSION_SCHEDULE_DIR=.cache/schedules
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

RFdiffusion 原始代码和权重请遵守其上游项目许可证、模型权重使用条款以及相关数据源要求。本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

如果在科研工作中使用 RFdiffusion 结果，建议引用 RFdiffusion 原始论文和 OneScience 相关项目信息，并根据实际任务补充下游结构设计、评估或可视化工具引用。
