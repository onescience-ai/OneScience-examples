<p align="center">
  <strong>
    <span style="font-size: 30px;">SA-PINNs</span>
  </strong>
</p>

# 模型介绍

SA-PINNs（Self-Adaptive Physics-Informed Neural Networks）为观测点、边界点和 PDE 配点引入可学习的正注意力权重，使模型在训练中自动关注难拟合区域。

本模型包提供一维 Laplace、二维 Helmholtz 和一维时变 Burgers 方程案例。

论文：Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism  
https://arxiv.org/abs/2009.04544

# 模型描述

SA-PINNs 使用 min-max 鞍点优化：网络参数通过梯度下降减小加权物理损失，注意力参数通过梯度上升提高难拟合位置的权重。注意力权重由归一化 `exp(alpha)` 构造，始终为正且均值接近 1；Adam 阶段同时更新网络和注意力，L-BFGS 阶段冻结注意力并精调网络。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 一维 Laplace 求解 | 观察注意力如何聚焦高残差配点。 |
| 二维 Helmholtz 求解 | 求解具有高频空间变化的二维解析解。 |
| Burgers 方程求解 | 联合初值、边界和非线性 PDE 残差训练。 |
| 自适应权重研究 | 比较普通 PINN 和逐点 soft-attention 加权。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 进行训练。
- CPU 可用于小配置流程验证。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

### 下载模型包

```bash
modelscope download --model OneScience/SA-PINNs --local_dir ./SA-PINNs
cd SA-PINNs
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

三个案例均根据解析方程和随机采样生成观测点、边界点和 PDE 配点，不依赖外部数据文件。Burgers 案例不内置参考解，因此仅输出预测场和注意力分布。

### 训练

通过 `--case` 选择案例：

```bash
python scripts/train.py --case laplace
python scripts/train.py --case helmholtz
python scripts/train.py --case burgers
```

检查点和训练历史默认保存至 `weight/` 和 `result/`。命令行参数可覆盖 `conf/config.yaml`，例如 `--epochs`、`--lbfgs-iters`、`--n-pde` 和 `--device`。

### 训练权重

本仓库在`weight/`文件夹内提供三个案例训练的权重


### 推理、评估和可视化

完成对应案例训练后运行：

```bash
python scripts/inference.py --case laplace
python scripts/inference.py --case helmholtz
python scripts/inference.py --case burgers
```

预测结果、误差图和 PDE 注意力分布保存在 `result/` 目录。Laplace 和 Helmholtz 案例会输出相对 L2 误差，模型和训练默认参数可在 `conf/config.yaml` 中修改。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- McClenny, L. and Braga-Neto, U. Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism. arXiv:2009.04544, 2020.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文说明。
