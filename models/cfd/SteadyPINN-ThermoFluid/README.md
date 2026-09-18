<p align="center">
  <strong>
    <span style="font-size: 30px;">SteadyPINN-ThermoFluid</span>
  </strong>
</p>

# 模型介绍

本模型复现论文《Enhancing neural network extrapolation in thermo-fluid systems using steady-state solutions》(arXiv:2606.18417)。

论文提出 steady-state-informed 神经网络表示：将解分解为稳态分量 us(x) 与瞬态修正 g(x,t) 的乘积形式 u_h(x,t) = us(x) + f(t)·g(x,t)，其中 f(t) 是时间衰减函数（可训练参数）。当 f(t)→0 且 g 有界时，网络预测自动收敛到稳态解，从而把"解收敛到稳态"的渐进行为直接嵌入网络架构，而不是通过额外的罚项。该设计显著改善了神经网络在训练时间区间之外的时序外推能力。

本模型在 1D 热方程 benchmark（Ω=(0,1), α=0.1）上完成 Tier1 端到端训练验证：训练区间 t∈[0,0.5]，外推至 t=5。指数衰减 profile f(t)=e^{-λt} 下外推相对 L2 误差保持有界（t=5 时 ~4e-3），而常数 profile（f(t)=1，baseline）外推误差单调上升（t=5 时 ~0.31），充分验证了论文的核心主张。

# 模型描述

模型为 PyTorch 实现的 steady-state-informed PINN，包含四个全连接网络块（与论文 Figure 1 架构一致）：

- FNN1：空间坐标 x → 空间 embedding（[1, 50×2, 25]）
- FNN2：时间 t → 时间 embedding（[1, 50×2, 25]）
- FNN3：空间 embedding → 稳态场 us,φ(x)（[25, 50×3, 1]，输出层 linear）
- FNN4：空间+时间 embedding 拼接 → 瞬态修正 gθ(x,t)（[50, 50×3, 1]，输出层 tanh 限幅 g∈(-1,1)）

- 核心实现文件：`model/model.py`（FNN1-4 + ansatz）、`model/temporal.py`（四种 f(t) profile：constant / exponential / algebraic / damped oscillatory，含可训练参数与投影约束）、`model/soap.py`（SOAP 优化器，2D 权重矩阵用 SOAP + 1D 参数用 Adam）
- 参数量：22006
- 训练：两阶段（Phase 1 训 FNN1+FNN3 稳态并冻结；Phase 2 训 FNN2+FNN4 瞬态），SOAP 优化器（η=1e-3, np=10, β1=0.9, β2=0.999, ε=1e-8），PINN 复合损失（残差+边界+初始，权重默认 1）

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| PDE 时序外推 | 训练区间内拟合、训练区间外（外推）保持正确渐近收敛到稳态 |
| thermo-fluid 代理建模 | 耗散性 PDE（热方程、热-流耦合）长期瞬态演化预测 |
| 模型训练 | 使用两阶段训练策略 + SOAP 优化器训练 steady-state-informed PINN |
| 模型推理 | 加载权重，对任意 (x,t) 前向输出解场 |

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
modelscope download --model OneScience/SteadyPINN-ThermoFluid --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem、bio、all（全领域）
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本模型使用解析合成数据：1D 热方程 ∂u/∂t - α∂²u/∂x² = γ(x)，Ω=(0,1)，α=0.1。边界 u(0,t)=u(1,t)=0，初始 u(x,0)=sin(πx)，稳态 us(x)=x(1-x)(3+sin(3πx)+sin(2πx))。参考解通过本征模展开解析计算，无需下载外部数据集。

### 训练

单卡：

```bash
python scripts/run_train.py --config conf/heat_eq_tier1.yaml --device dcu --seed 0
```

训练采用两阶段：Phase 1 训稳态网络（FNN1+FNN3），Phase 2 冻结稳态网络并训练瞬态网络（FNN2+FNN4）。训练会在 `outputs/checkpoints/` 下保存 `final_model.pt`。

### 训练权重

已上传权重：`weight/final_model.pt`（steady-state-informed PINN，1D heat，exponential profile，Tier1 训练收敛：Phase1 loss 6.7e-6，Phase2 loss 2.9e-6，学习 λ≈1.02）。

### 推理

```bash
python scripts/run_evaluate.py --config conf/heat_eq_tier1.yaml --checkpoint weight/final_model.pt --device dcu
```

推理输出各时间点的相对 L2 误差（区间内 t=0.1/0.3/0.5 与 外推 t=1/2/5），保存至 `outputs/metrics/evaluate_metrics.json`。

### 评估和可视化

```bash
python scripts/run_smoke.py --device cpu
```

冒烟测试（Tier0 6 项检查：forward/backward/train-loop/val-loop/外推评估/config 一致性）可用于验证环境与代码正确性。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为论文《Enhancing neural network extrapolation in thermo-fluid systems using steady-state solutions》(arXiv:2606.18417) 的复现版本。
- 作者：Sanjeeb Poudel (Florida State University), Teeratorn Kadeethum (Siemens Energy), Sanghyun Lee (Florida State University)
- 论文链接：https://arxiv.org/abs/2606.18417

