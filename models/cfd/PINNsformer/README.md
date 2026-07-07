# PINNsformer

<p align="center">
  <strong>
    <span style="font-size: 20px;">点击下方图片，体验一键式 PINNsformer 模型开发</span>
  </strong>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/OneScience/PINNsformer" target="_blank" rel="noopener noreferrer">
    <img src="https://www.modelscope.cn/api/v1/models/VoyagerX/OneScience-badge/repo?Revision=master&FilePath=LOGOs.png" width="200" alt="Logo">
  </a>
</p>

## OneScience 官方信息

| 平台 | 文档 | OneScience 主仓库 | Skills 仓库 |
|---|---|---|---|
| Gitee | https://gitee.com/onescience-ai/onescience-doc | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience-doc | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

## 项目说明

PINNsformer 是一种基于 Transformer 的物理信息神经网络，用多头注意力机制和时空嵌入捕捉偏微分方程中的时间依赖关系。它将点式 PINN 输入转换为伪序列，并使用序列损失来改进传统 MLP PINN 在全局传播初值约束和拟合时空场时的表现。

本仓库是 OneScience 标准运行包，包含一维反应方程、一维波动方程、对流方程和 Navier-Stokes 圆柱绕流四类示例，并保留 PINNsformer、经典 PINN、QRes、FLS 和部分 NTK 对比脚本。`convection` 与 `navier_stokes` 示例需要 `.mat` 数据文件；本标准包按原始 README 的 `data.sh` 逻辑准备数据。

## Resource Card

| 字段 | 内容 |
|---|---|
| 资源类型 | 模型 |
| OneScience 领域 | cfd |
| 领域标签 | cfd, pde, physics_informed_neural_network, transformer |
| 任务 | pde_solution_approximation |
| 任务标签 | train, inference, evaluation, visualization |
| 主平台资源 | https://modelscope.cn/models/OneScience/PINNsformer |
| 标准运行包工作目录 | `.` |
| OneScience examples 兼容路径 | `onescience/examples/cfd/PINNsformer` |
| 必需模型文件 | `1d_reaction/*.py`, `1d_wave/*.py`, `convection/*.py`, `navier_stokes/*.py`, `data.sh`, `scripts/preflight_pinnsformer.py` |
| 数据准备 | `bash data.sh` 将 `convection.mat` 和 `cylinder_nektar_wake.mat` 放入对应子目录 |
| 支持能力 | 预检、训练、评测、可视化 |
| 最小验证 | `python scripts/preflight_pinnsformer.py --repo-root .` |

能力和命令要求

| 能力 | 必须提供 |
|---|---|
| `preflight` | `python scripts/preflight_pinnsformer.py --repo-root .`，检查代码入口、数据文件位置和 Manifest 约束 |
| `prepare` | `bash data.sh`，从统一数据目录复制 `convection.mat` 和 `cylinder_nektar_wake.mat` |
| `train` | 在对应子目录执行训练脚本，例如 `cd 1d_reaction && python 1d_reaction_pinnsformer.py` |
| `inference` | 训练脚本完成后会同步生成预测结果图，可视为示例推理与可视化流程 |
| `evaluate` | 脚本内置与解析解或真实场的误差/可视化对比 |
| `visualize` | 输出到各子目录的 `result/*.png` |
| `deploy` | 暂未声明 |

## 文件说明

| 路径 | 类型 | 作用 | 是否必需 | 用于能力 | 下载后放置位置 | 备注 |
|---|---|---|---|---|---|---|
| `README.md` | 文档 | 人类用户和大模型入口，说明文件、下载、预检和运行方式 | 是 | 全部能力 | 模型包根目录 | 中文正文 |
| `onescience_run_manifest.yaml` | Manifest 文件 | 机器可读运行说明，声明资源 ID、命令、输出和诊断 | 是 | 全部能力 | 模型包根目录 | 大模型必须解析 |
| `manifest.yaml` | Manifest 兼容入口 | 指向同一套运行协议，便于默认路径发现 | 是 | 全部能力 | 模型包根目录 | 内容与运行 Manifest 保持一致 |
| `scripts/preflight_pinnsformer.py` | 预检脚本 | 检查目录结构、脚本文件、可选数据文件和 `.mat` schema | 是 | 预检 | `scripts/preflight_pinnsformer.py` | 不启动训练 |
| `data.sh` | 数据准备脚本 | 按原始 README 逻辑复制两个 `.mat` 数据文件 | 是 | prepare, train | `data.sh` | 不下载到数据集仓库 |
| `1d_reaction/*.py` | 训练脚本 | 一维反应方程的 PINNsformer/PINN/QRes/FLS 对比 | 是 | train, visualize | `1d_reaction/` | 不依赖外部数据 |
| `1d_wave/*.py` | 训练脚本 | 一维波动方程的 PINNsformer/PINN/NTK 对比 | 是 | train, visualize | `1d_wave/` | 不依赖外部数据 |
| `convection/*.py` | 训练脚本 | 对流方程的 PINNsformer/PINN/QRes/FLS 对比 | 是 | train, visualize | `convection/` | 需要 `convection/convection.mat` |
| `navier_stokes/*.py` | 训练脚本 | Navier-Stokes 圆柱绕流的 PINNsformer/PINN/QRes/FLS 对比 | 是 | train, visualize | `navier_stokes/` | 需要 `navier_stokes/cylinder_nektar_wake.mat` |
| `run_pinnsformer.ipynb` | Notebook | Navier-Stokes 交互式示例 | 否 | 示例 | 模型包根目录 | 供交互使用 |

## Manifest

机器可读 Manifest 位于仓库根目录 `onescience_run_manifest.yaml`，并提供 `manifest.yaml` 作为兼容入口。修改模型入口、下载命令、数据准备方式、运行矩阵或输出路径后，必须同步更新这两个文件，并建议执行 YAML 解析和 `command_refs` 校验。

## 模型 vs 数据集关系

本次整理按用户要求不生成独立数据集标准仓库，也不声明 ModelScope 数据集仓库 ID。`convection` 和 `navier_stokes` 两类示例仍需要原 README 中的数据文件：

| 示例 | 数据文件 | 默认放置位置 | 准备方式 |
|---|---|---|---|
| 对流方程 | `convection.mat` | `convection/convection.mat` | `bash data.sh` |
| Navier-Stokes | `cylinder_nektar_wake.mat` | `navier_stokes/cylinder_nektar_wake.mat` | `bash data.sh` |

`data.sh` 默认从 `${ONESCIENCE_DATASETS_DIR}/pinnsformer` 复制数据；如果未设置该环境变量，则使用 OneScience 环境文件中的统一数据目录 `/public/share/sugonhpcapp01/onestore/onedatasets/pinnsformer`。也可以根据原始 README 从百度网盘下载后手动放到上述路径。

原始 README 提供的数据集下载方式：

| 下载源 | 链接 | 提取码 | 下载后放置位置 |
|---|---|---|---|
| 百度网盘 | https://pan.baidu.com/s/1pM4ICc6FJX5pLF7WEoozxQ?pwd=5gha | `5gha` | `convection/convection.mat` 和 `navier_stokes/cylinder_nektar_wake.mat` |

## 文件与下载

下载模型：

```bash
modelscope download --model OneScience/PINNsformer
```

如果使用 `--cache_dir` 下载，请先 `cd` 到实际下载后的模型包根目录再执行运行命令：

```bash
cd /path/to/downloaded/OneScience_PINNsformer
```

准备数据：

```bash
bash data.sh
```

如当前环境没有统一数据目录，可使用原始 README 提供的百度网盘链接下载数据集：

- 百度网盘：`https://pan.baidu.com/s/1pM4ICc6FJX5pLF7WEoozxQ?pwd=5gha`
- 提取码：`5gha`

如需手动准备数据，请放置为：

```text
convection/convection.mat
navier_stokes/cylinder_nektar_wake.mat
```

## 环境安装

```bash
bash install.sh cfd
```

还需要运行环境中可导入 `torch`、`numpy`、`scipy`、`matplotlib` 和 `pyyaml`。若只执行结构预检，可先不安装深度学习依赖。

## 运行流程

### 1. 环境预检

```bash
python - <<'PY'
import torch, numpy, scipy, matplotlib
print("environment ok")
PY
```

### 2. 下载模型包

```bash
modelscope download --model OneScience/PINNsformer
cd /path/to/downloaded/OneScience_PINNsformer
```

### 3. 准备数据

```bash
bash data.sh
```

### 4. 运行前预检

```bash
python scripts/preflight_pinnsformer.py --repo-root .
```

成功时应看到 `[OK] model preflight completed`。如需检查 `.mat` 文件可读性和关键变量：

```bash
python scripts/preflight_pinnsformer.py --repo-root . --check-data-schema
```

### 5. 运行

一维反应方程 PINNsformer：

```bash
cd 1d_reaction
python 1d_reaction_pinnsformer.py
```

一维波动方程 PINNsformer：

```bash
cd 1d_wave
python 1d_wave_pinnsformer.py
```

对流方程 PINNsformer：

```bash
cd convection
python convection_pinnsformer.py
```

Navier-Stokes PINNsformer：

```bash
cd navier_stokes
python navier_stoke_pinnsformer.py
```

### 6. 验证输出

各脚本会在当前子目录生成 `model/` 和 `result/`。模型权重保存为 `model/*.pt`，可视化结果保存为 `result/*.png`。

## 输出说明

| 输出路径 | 说明 | 成功标准 |
|---|---|---|
| `1d_reaction/model/*.pt` | 一维反应方程训练权重 | 训练脚本完成后生成 |
| `1d_reaction/result/*.png` | 一维反应方程预测图 | 训练脚本完成后生成 |
| `1d_wave/model/*.pt` | 一维波动方程训练权重 | 训练脚本完成后生成 |
| `1d_wave/result/*.png` | 一维波动方程预测图 | 训练脚本完成后生成 |
| `convection/model/*.pt` | 对流方程训练权重 | 训练脚本完成后生成 |
| `convection/result/*.png` | 对流方程预测图 | 训练脚本完成后生成 |
| `navier_stokes/model/*.pt` | Navier-Stokes 训练权重 | 训练脚本完成后生成 |
| `navier_stokes/result/*.png` | Navier-Stokes 预测图 | 训练脚本完成后生成 |

## 预检与诊断

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `missing required source file` | 模型包下载不完整或工作目录错误 | `cd` 到模型包根目录，重新下载 `OneScience/PINNsformer` |
| `missing data file` | 未执行 `bash data.sh` 或数据目录不正确 | 设置 `ONESCIENCE_DATASETS_DIR` 后执行 `bash data.sh`，或手动放置两个 `.mat` 文件 |
| `source data not found` | 统一数据目录中没有 PINNsformer 数据 | 根据原 README 从百度网盘下载，或确认 `/public/share/sugonhpcapp01/onestore/onedatasets/pinnsformer` 是否存在 |
| `ModuleNotFoundError: scipy` | 未安装 SciPy，无法读取 `.mat` 文件 | 安装 OneScience CFD 环境或补充 `scipy` |
| `No such file or directory: './model/...'` | 子目录下没有 `model/` 或 `result/` | 训练脚本会自动创建；若失败，确认当前目录可写 |
| `CUDA out of memory` | 训练迭代和采样规模较大，显存不足 | 使用更大显存设备，或在确认实验允许后调整脚本参数 |

## 限制与适用范围

本标准包保留原始示例脚本中的训练轮数、采样数和网络结构，没有改写为统一 YAML 配置。`1d_reaction` 与 `1d_wave` 可不依赖外部数据完成训练；`convection` 与 `navier_stokes` 必须先准备 `.mat` 数据文件。

## 引用与许可证

论文：`https://arxiv.org/abs/2307.11833`。当前整理包未在原始目录中发现明确许可证文件，公开分发前建议补充许可证信息。
