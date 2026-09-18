<p align="center">
  <strong>
    <span style="font-size: 30px;">DiffFluid-DDM</span>
  </strong>
</p>

# 模型介绍

DiffFluid-DDM 是 DiffFluid 论文（DiffFluid: Plain Diffusion Models are Effective Predictors of Flow Dynamics，arXiv:2409.13665）的复现实现。它将流体方程求解建模为条件去噪扩散生成（conditional DDPM + Diffusion Transformer），以朴素扩散模型配合 Transformer 结构高效求解流体动力学方程，覆盖 Navier-Stokes、Darcy flow 与翼型 Euler 方程三个 CFD 基准。通过 Multi-Resolution Noise 与 Multi-Loss（MSE+L1）策略显著提升求解精度。

论文：DiffFluid: Plain Diffusion Models are Effective Predictors of Flow Dynamics
https://arxiv.org/abs/2409.13665

# 模型描述

DiffFluid-DDM 基于 Diffusion Transformer（DiT）架构 + adaLN-Zero 时间条件注入：
- 前向加噪：Multi-Resolution Noise（多尺度高斯噪声叠加）+ Annealing 退火（Algorithm 1）
- 模型主干：Patch Embedding → DiT Block（adaLN-Zero 自注意力 + MLP）→ LayerNorm+Linear 输出头
- 训练目标：DDPM 噪声预测，MSE + L1 多损失
- 推理：从标准高斯采样，按线性调度逐步去噪生成物理场
- 支持 Navier-Stokes（64×64 网格）与 Darcy flow（85×85 网格）两个 benchmark 的端到端求解

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| CFD 场预测 | 从条件输入生成流体方程解（NS 未来 10 步涡量 / Darcy 压力场） |
| 条件去噪扩散求解 | 以多分辨率噪声训练 + 标准 DDPM 推理的通用 PDE 求解 |
| 本地快速验证 | 使用小数据子集检查数据读取、模型训练、推理与评估 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/DiffFluid-DDM --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

训练数据为 FNO benchmark（论文 [4] 引用）：
- Navier-Stokes：`NavierStokes_V1e-5_N1200_T20.mat`（涡量场，1200 样本，64×64 网格，前 10 步→后 10 步）
- Darcy：`piececonst_r421_N1024_smooth2.mat`（多孔介质结构系数→压力场，1024 样本，下采样至 85×85）

（请在此处说明训练数据来源和获取方式）

### 训练

单卡：

```bash
python scripts/train.py --config conf/diff_fluid.yaml
python scripts/train.py --config conf/diff_fluid_darcy.yaml
```

训练会在 `checkpoints/` 与 `checkpoints_darcy/` 下保存 `best.pt` / `last.pt`。

### 训练权重

- `weight/best.pt`：Navier-Stokes 模型权重（200 样本子集训练，test rel-L2 = 7.26）
- `weight/last.pt`：Darcy 模型权重（200 样本子集训练，test rel-L2 = 0.45）

### 推理

```bash
python scripts/infer.py --config conf/diff_fluid.yaml --checkpoint weight/best.pt
python scripts/infer.py --config conf/diff_fluid_darcy.yaml --checkpoint weight/last.pt
```

推理结果会保存至 `outputs/`（predictions.npy / targets.npy）。

### 评估和可视化

```bash
python scripts/evaluate.py --config conf/diff_fluid.yaml --checkpoint weight/best.pt
python scripts/evaluate.py --config conf/diff_fluid_darcy.yaml --checkpoint weight/last.pt
```

评估结果（相对 L2 error）写入 `logs/metrics.json` / `logs_darcy/metrics.json`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

开源仓库复用上游内容，复现的论文可参考下面描述

- 本仓库为 DiffFluid 原始论文的复现版本：DiffFluid: Plain Diffusion Models are Effective Predictors of Flow Dynamics（arXiv:2409.13665）。

