<p align="center">
  <strong>
    <span style="font-size: 30px;">GNODE</span>
  </strong>
</p>

# 模型介绍

GNODE（Graph Neural Ordinary Differential Equations）是 arXiv:2607.18309 提出的非定常翼型气动时空预测框架：以 GNN（Graph Network Simulator 结构）作为向量场，结合增广隐变量维度与 Neural ODE（RK4 积分），在外源控制（俯仰角时序）作用下对 RAE 2822 俯仰翼型的表面量（cp、cf,x、cf,z）做连续时间时空预测。相比自回归 GNN 基线（GNS），GNODE 时序更稳定、空间更平滑、精度更高，能缓解自回归误差累积与相位滞后。

论文：Spatio-Temporal Prediction of Unsteady Airfoil Aerodynamics Using Augmented Graph Neural Ordinary Differential Equations with Exogenous Controls
https://arxiv.org/abs/2607.18309

# 模型描述

- 架构：encoder-processor-decoder 图神经网络（GNS 结构）作为向量场 + 增广隐变量 L（14 维/节点）+ RK4 四阶龙格库塔 ODESolve + 解析外源控制 f_u(t)。
- 输入：初始表面流场 Y0 (512×3)、静态几何 G (512×4)、控制信号 u(t)（α, dα/dt）、图结构。
- 输出：表面流场时间序列 Y_t (T×512×3)。
- 训练损失：MSE(Y_truth, Y_pred) + λ‖L‖²（λ=2.23e-6）。
- 权重：best_model.pt。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 模型训练 | 使用合成 RAE 2822 俯仰翼型数据训练 GNODE（32 训练轨迹 / 8 测试轨迹） |
| 模型推理 | 加载权重进行时空 rollout 预测 |
| 评估 | 计算表面 MAE/MSE/R² 与全局 cmy 指标 |
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
modelscope download --model OneScience/GNODE --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
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

本复现使用合成 RAE 2822 俯仰翼型数据（论文原始 URANS 数据需向作者申请，数据生成器见 `data/synthetic_dataset.py`，DoE：α0∈(0.0, 1.5, 2.79, 3.5)° × k∈(0.1, 0.2, 0.3, 0.5, 1.0) × α̂∈(1.0, 2.0)°）。

### 训练

```bash
python scripts/train.py
```

训练会保存 best checkpoint 到 `checkpoints/` 下。

### 训练权重

- best_model.pt（GNODE 主模型，90,757 参数，best val_mae=0.248）
- best_model_gns.pt（GNS 基线，504,406 参数，best val_mae=0.592）

### 推理

```bash
python model/roll_out.py
```

推理结果会保存至 `outputs/`。

### 评估和可视化

```bash
python scripts/evaluate.py
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

开源仓库复用上游内容，复现的论文可参考下面描述

- 本仓库为 arXiv:2607.18309（GNODE for Unsteady Airfoil Aerodynamics）的复现版本。

