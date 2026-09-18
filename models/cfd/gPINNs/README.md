<p align="center">
  <strong>
    <span style="font-size: 30px;">gPINNs</span>
  </strong>
</p>

# 模型介绍

gPINNs（Gradient-enhanced Physics-Informed Neural Networks）在传统 PINNs 的 PDE 残差约束之外，进一步约束残差对输入坐标的梯度。额外的高阶导数信息可提高训练数据利用率，并改善部分正问题和反问题的求解精度。

本模型包提供一维 Poisson、二维 Poisson 和带残差自适应加密（RAR）的 Burgers 方程案例。

论文：Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems  
https://doi.org/10.1016/j.cma.2022.114823

# 模型描述

gPINNs 使用自动微分同时计算 PDE 残差及其坐标梯度，并将两者作为训练约束。二维 Poisson 案例使用硬边界输出变换；Burgers 案例使用 RAR 机制，在高残差区域逐步补充配点。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 一维 Poisson 求解 | 验证残差梯度约束和边界损失的组合训练流程。 |
| 二维 Poisson 求解 | 使用硬边界输出变换求解零 Dirichlet 边界问题。 |
| Burgers 方程求解 | 使用高阶自动微分和 RAR 加密高残差区域。 |
| 模型流程验证 | 使用随包数据和预训练权重检查训练、推理和绘图流程。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 进行训练。
- CPU 可用于推理和小配置流程验证，完整训练速度较慢。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

### 下载模型包

```bash
modelscope download --model OneScience/gPINNs --local_dir ./gPINNs
cd gPINNs
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

一维和二维 Poisson 案例使用解析解构造训练与评估数据；Burgers 案例使用随包的参考解 `data/Burgers.npz`。数据和训练参数均可在 `conf/config.yaml` 中调整。

### 训练

训练全部案例：

```bash
python scripts/train.py --case all
```

也可通过 `--case 1d`、`--case 2d` 或 `--case burgers` 单独训练。`--epochs` 和 `--nf` 可覆盖所选案例的默认配置，`--lbfgs-iters` 仅用于一维 Poisson，`--rar-rounds` 用于调整 Burgers 案例的 RAR 轮数，`--quick` 可跳过 RAR 以快速验证流程。

### 训练权重

本模型包在 `weight/` 文件夹内提供一维 Poisson、二维 Poisson 和 Burgers 方程案例的预训练权重。

### 推理、评估和可视化

模型包提供用于流程验证的三个案例权重，可直接运行：

```bash
python scripts/inference.py --case all
```

脚本会输出各案例的相对 L2 误差，并在 `result/` 下保存包含预测结果的可视化图。可通过 `--case` 指定单个案例；模型、数据和训练参数均可在 `conf/config.yaml` 中修改。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Yu, J., Lu, L., Meng, X., and Karniadakis, G. E. Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems. Computer Methods in Applied Mechanics and Engineering, 393, 114823, 2022.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
