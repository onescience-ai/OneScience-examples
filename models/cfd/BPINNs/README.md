<p align="center">
  <strong>
    <span style="font-size: 30px;">BPINNs</span>
  </strong>
</p>

# 模型介绍

BPINNs（Bayesian Physics-Informed Neural Networks）将物理方程约束引入贝叶斯神经网络，用于在含噪数据下同时估计方程解和预测不确定性。本模型包提供一维 Laplace 方程求解案例：

```text
u_xx + pi^2 sin(pi x) = 0,  x in [0, 1]
u(x) = sin(pi x)
```

论文：B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data  
https://doi.org/10.1016/j.jcp.2020.109913

# 模型描述

BPINNs 联合观测数据、边界条件和 PDE 残差进行训练，默认使用 Adam 优化并通过 L-BFGS 进一步精调。推理时可从检查点中的多个参数状态计算预测均值和标准差；默认训练只生成一个优化状态，如需完整贝叶斯不确定性估计，需提供 HMC、变分推断或集成采样得到的后验参数状态。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 一维 Laplace 求解 | 使用观测点、边界点和配点共同训练方程解。 |
| 含噪 PDE 建模 | 在物理约束下拟合带噪观测数据。 |
| 优化器对比 | 比较 Adam 与 L-BFGS 对 PINN 收敛的影响。 |
| 后验预测 | 从多个参数状态计算预测均值和标准差。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- GPU 或 DCU 可加快正式训练；CPU 可用于流程验证。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

### 下载模型包

```bash
modelscope download --model OneScience/BPINNs --local_dir ./BPINNs
cd BPINNs
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

当前案例根据解析解自动生成域内观测点、边界点、PDE 配点和测试网格，不依赖外部数据文件。可在 `conf/config.yaml` 中修改观测点数量、配点数量、噪声强度及计算域配置。

### 训练

```bash
python scripts/train.py
```

默认生成基础权重 `weight/bpinn_laplace1d.pt`，训练历史保存至 `result/training_history.npz`。

### 训练权重

本仓库在`weight/`文件夹内提供基于一维 Laplace方程训练的权重。

### L-BFGS 精调

```bash
python scripts/refine.py
```

精调后默认生成 `weight/bpinn_laplace1d_refined.pt`。

### 推理、评估和可视化

```bash
python scripts/inference.py
```

推理优先读取精调权重，不存在时读取基础权重，并将预测结果和可视化图保存至 `result/`。模型、训练、损失和推理参数均可在 `conf/config.yaml` 中修改。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Yang, L., Meng, X., and Karniadakis, G. E. B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data. Journal of Computational Physics, 425, 109913, 2021.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文说明。
