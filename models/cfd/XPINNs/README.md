<p align="center">
  <strong>
    <span style="font-size: 30px;">XPINNs</span>
  </strong>
</p>

# 模型介绍

XPINNs（Extended Physics-Informed Neural Networks）使用域分解方法将复杂计算区域拆分为多个子域，并为每个子域配置独立的神经网络。训练时除约束 PDE 残差和外边界条件外，还约束子域接口处的解连续性和残差一致性。

本模型包复现 XPINNs 论文中的二维 Poisson 基准案例：X 形不规则区域被拆分为三个子域，控制方程为：

```text
u_xx + u_yy = exp(x) + exp(y)
```

论文：Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations  
https://doi.org/10.4208/cicp.OA-2020-0164

# 模型描述

XPINNs 为三个子域分别配置神经网络，默认使用 `tanh`、`sin` 和 `cos` 激活函数。模型通过联合优化边界损失、PDE 残差、接口值和接口残差损失，实现在不规则区域上的方程求解。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 复杂区域 PDE 求解 | 在 X 形不规则区域上求解二维 Poisson 方程。 |
| 域分解研究 | 为不同子域配置独立网络结构和激活函数。 |
| 接口约束研究 | 比较接口解连续性与残差一致性损失。 |
| 模型流程验证 | 使用随包数据和小规模配置检查训练、推理流程。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 进行训练。
- CPU 可用于小配置流程验证，完整采样训练速度较慢。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

### 下载模型包

```bash
modelscope download --model OneScience/XPINNs --local_dir ./XPINNs
cd XPINNs
```

### 安装运行环境

**DCU 环境**

```bash
# 请首先激活 DTK 及 CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
# 请首先激活 CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

模型包内置二维 Poisson 数据文件 `data/XPINN_2D_PoissonEqn.mat`，其中包含三个子域的内部点、边界点、接口点和精确解。可在 `conf/config.yaml` 中调整各类训练样本数量。

### 训练

```bash
python scripts/train.py
```

默认权重保存至 `weight/xpinn_poisson_2d.pt`。

### 训练权重

本仓库在 `weight/` 文件夹内提供基于二维 Poisson 数据集训练的权重。

### 推理、评估和可视化

完成训练后运行：

```bash
python scripts/inference.py
```

脚本会输出整体相对 L2 误差，并将精确解、预测结果和绝对误差图保存至 `result/xpinn_poisson_2d.png`。模型、数据、损失和推理参数均可在 `conf/config.yaml` 中修改。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Jagtap, A. D., Kharazmi, E., and Karniadakis, G. E. Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations. Communications in Computational Physics, 28(5), 2002-2041, 2020.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
