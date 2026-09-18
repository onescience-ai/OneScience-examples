<p align="center">
  <strong>
    <span style="font-size: 30px;">SpectraNet</span>
  </strong>
</p>

# 模型介绍

SpectraNet 是 arXiv:2605.09096《Bridging Spectral Operator Learning and U-Net Hierarchies: SpectraNet for Stable Autoregressive PDE Surrogates》的复现实现。它是一个自回归神经算子（autoregressive neural operator），将截断谱卷积（truncated spectral convolution）组合进 U-Net 层级，配合 Residual-Target Spectral Block 与 Semigroup-Consistency Loss 训练，用于时变 PDE 的稳定自回归代理建模。在 Navier-Stokes ν=10^-5 64×64 基准上，论文报告 test relative L2=0.0822（2.04M 参数），比 FNO 少 2.33× 参数且误差低约 20%，并在 T=100 自由 rollout 中保持有界（FNO 发散）。

本复现包使用小规模数据（Tier 1，100 轨迹训练）端到端验证了架构与训练协议的正确性：test joint-trajectory relative L2=0.1647，参数 2,040,705，T=100 长程 rollout 0/40 发散。

# 模型描述

- 输入：`(B, 64, 64, Tin=10)` 涡量窗口 + 归一化 (x,y) 网格（12 通道）
- 输出：`(B, 64, 64, 1)` 下一帧（残差 Δ_t，集成后 ŵ_{t+1}=ω_t+Δ_t）
- 主干：三层 encoder-bottleneck-decoder U-Net（channels {32,64,128,128}，截断谱模式 {12,6,3,1}）
- 关键组件：TruncatedSpectralConv2d（rfft2/irfft2 + 双 MxM 块截断 + 复权重 einsum）、SpectralBlock（Spec+MLP1x1+Conv1x1+GeLU）、Residual-Target 参数化、Semigroup-Consistency Loss（λ=0.1）
- 主要文件：`model/model.py`（模型）、`model/spectral.py`（截断谱卷积）、`model/blocks.py`（编码/解码块）、`model/losses.py`（损失）、`model/data.py`（数据管线）

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 模型训练 | 使用 NavierStokes V1e-5 涡量数据训练 SpectraNet（自回归 + Semigroup-Consistency Loss） |
| 模型推理 | 加载权重对涡量窗口做自由自回归 rollout 预测 |
| 模型评估 | 计算 joint-trajectory relative L2 与长程 rollout 稳定性 |
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
modelscope download --model OneScience/SpectraNet --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install torch numpy scipy pyyaml -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install torch numpy scipy pyyaml
```

### 训练数据介绍

训练数据为 NavierStokes V1e-5 N1200 T20.mat（FNO 作者公开数据，1200 轨迹 × 20 时间步 × 64×64 涡量，Li et al. 2021）。数据路径在 `conf/spectranet_ns_v1e5.yaml` 中通过 `data.mat_path` 配置，请按实际存放位置修改后训练。

### 训练

```bash
python scripts/train.py --config conf/spectranet_ns_v1e5.yaml
```

训练会在 `outputs/tier1/checkpoints/` 下保存 best-validation checkpoint `spectranet_best.pt`（AdamW + OneCycleLR，Residual-Target + Semigroup-Consistency Loss）。

### 训练权重

- `weight/spectranet_best.pt`：Tier 1 小数据训练 checkpoint（100 轨迹 × 200 epochs，best_val_l2=0.1688，test rel L2=0.1647，2,040,705 参数）

### 推理

```bash
python scripts/infer.py --config conf/spectranet_ns_v1e5.yaml --ckpt weight/spectranet_best.pt --steps 10
```

自由自回归 rollout，输出预测轨迹保存至 `outputs/rollout_pred.pt`。

### 评估和可视化

```bash
python scripts/eval.py --config conf/spectranet_ns_v1e5.yaml --ckpt weight/spectranet_best.pt
```

计算 test joint-trajectory relative L2、参数计数与 T=100 长程 rollout 稳定性，结果写入 `outputs/tier1/test_results.json`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为以下论文的复现版本（OneScience 复现实现，未使用官方代码仓库）：

- Bridging Spectral Operator Learning and U-Net Hierarchies: SpectraNet for Stable Autoregressive PDE Surrogates, arXiv:2605.09096, https://arxiv.org/abs/2605.09096

