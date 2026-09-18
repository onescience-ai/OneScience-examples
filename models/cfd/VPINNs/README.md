<p align="center">
  <strong>
    <span style="font-size: 30px;">hp-VPINN</span>
  </strong>
</p>

# 模型介绍

hp-VPINN（Variational Physics-Informed Neural Network with h- and p-Refinement）使用偏微分方程的变分弱形式训练神经网络。模型通过测试函数对方程残差施加积分约束，并结合域分解（h-refinement）和高阶多项式测试函数（p-refinement）提升复杂解的表达和求解能力。

本模型包提供一维和二维 Poisson 方程案例。

论文：hp-VPINNs: Variational physics-informed neural networks with domain decomposition  
https://doi.org/10.1016/j.cma.2020.113547

# 模型描述

hp-VPINN 使用 Gauss-Lobatto-Jacobi 数值积分构造变分损失，并以全连接网络表示方程解。一维案例支持分段积分域，二维案例支持非方形子域网格；可分别调整子域数量和测试函数阶数，研究 h-、p-refinement 对求解效果的影响。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 一维 Poisson 求解 | 使用分段积分域和高阶测试函数约束弱形式残差。 |
| 二维 Poisson 求解 | 在二维域分解网格上执行张量积数值积分。 |
| h-refinement 研究 | 调整子域数量，比较空间分解效果。 |
| p-refinement 研究 | 调整测试函数阶数，比较弱形式约束能力。 |

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
modelscope download --model OneScience/VPINNs --local_dir ./VPINNs
cd VPINNs
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

一维和二维 Poisson 案例均使用解析解构造源项、边界值和评估数据，不依赖外部数据文件。可在 `conf/config.yaml` 中配置子域数量、测试函数数量和数值积分点数量。

### 训练

训练案例由 `conf/config.yaml` 中的 `common.case` 控制，支持 `1d`、`2d` 和 `all`：

```bash
python scripts/train.py
```

训练权重默认保存至 `weight/` 目录。

### 训练权重

本仓库在`weight/`文件夹内提供基于一维和二维 Poisson 数据集训练的权重。

### 推理、评估和可视化

模型包提供一维 Poisson 验证权重，可直接运行：

```bash
python scripts/inference.py
```

二维案例需先训练对应权重，再将 `common.case` 设为 `2d` 或 `all` 后运行推理。脚本会输出相对 L2 误差，并将结果图保存至 `result/`。模型、训练和推理参数均可在 `conf/config.yaml` 中修改。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Kharazmi, E., Zhang, Z., and Karniadakis, G. E. hp-VPINNs: Variational physics-informed neural networks with domain decomposition. Computer Methods in Applied Mechanics and Engineering, 374, 113547, 2021.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文说明。
