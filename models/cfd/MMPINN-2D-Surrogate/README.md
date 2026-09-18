<p align="center">
  <strong>
    <span style="font-size: 30px;">MMPINN-2D-Surrogate</span>
  </strong>
</p>

# 模型介绍

MMPINN-2D-Surrogate 是基于多量级损失（multi-magnitude loss）正则化 PINN 的二维偏微分方程
代理求解模型，采用多尺度傅里叶特征（Multi-scale Fourier Feature, MFF）网络，面向二维
multi-frequency 热传导方程开展快速场预测。模型以 `loss = (L_IC+BC)^(1/3) + (L_res)^(1/3)`
平衡监督损失与残差损失，配合 Adam + 三段 L-BFGS 多级训练。

本仓库基于 OneScience 技能，独立复现了 MMPINN-2D-Surrogate 论文中的主实验。

论文：[A practical PINN framework for multi-scale problems with multi-magnitude loss terms](https://arxiv.org/abs/2308.06672)

# 模型描述

MMPINN-2D-Surrogate 复现论文 Section 4.5 的 MMPINN-MFF 方法，用于求解计算域 $(x,t)\in[-1,1]\times[0,1]$ 上满足 $u_t=u_{xx}+f(x,t)$、初始条件 $u(x,0)=0$ 和边界条件 $u(\pm1,t)=\sin(2\pi t)$ 的多频热传导问题，其解析解为 $u(x,t)=\sin\!\left(20\pi t/(1+9x^2)\right)$；模型将时空坐标归一化后通过标准差 $\sigma=10$ 的冻结傅里叶特征进行编码，并采用四层、每层 300 个神经元的 `tanh` 全连接网络预测标量场，同时以 $(L_{\mathrm{IC}}+L_{\mathrm{BC}})^{1/3}+L_{\mathrm{res}}^{1/3}$ 的多量级损失平衡初边值约束与 PDE 残差，从而提升对空间相关高频特征的建模精度。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 多尺度 PDE 求解 | 高频率/多量级损失占优的物理场代理求解 |
| PINN 加速 | 相比传统数值方法快速逼近解析解，替代大规模离散求解 |

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
modelscope download --model OneScience/MMPINN-2D-Surrogate --local_dir ./MMPINN-2D-Surrogate
cd MMPINN-2D-Surrogate
```

### 安装运行环境

**DCU 环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

```

### 训练数据介绍

本项目无需外部数据集，训练数据依据原始论文所定义的解析解 $u(x,t)=\sin\!\left(20\pi t/(1+9x^2)\right)$ 在计算域 $(x,t)\in[-1,1]\times[0,1]$ 内在线合成，包括 1,200 个初始条件点、分别施加于 $x=-1$ 和 $x=1$ 两侧边界的 1,200 个时间采样点，以及通过拉丁超立方采样生成的 120,000 个域内 PDE 残差配点；初始与边界样本以解析解值作为监督信号，域内配点则通过约束 $u_t-u_{xx}-f(x,t)=0$ 参与物理信息训练，另构建 $1200\times1200$ 规则网格及对应解析解，仅用于模型误差评估与结果可视化。

### 训练

```bash
python scripts/train.py --epochs 2000 --device cuda:0 --outdir outputs
```

训练协议（论文）：Adam 2000 iter (lr=1e-3) → 三段 L-BFGS (loss^(1/3), loss^(1/2), loss^1)。
收敛自动提前停（相对下降 <1e-5 持续 200 evals）。纯 Adam 参考：`--lbfgs 0`。

默认训练会保存 checkpoint：

```text
./outputs/model_2d.pth
./outputs/1.mat
```

### 训练权重

本仓库在 `weight/` 文件夹内提供基于锁种子训练导出的预训练模型权重，可用于直接推理。



### 推理

```bash
python scripts/inference.py --checkpoint weight/best_model.pt --outdir outputs
```

输出：`outputs/1.mat`（预测场 `u`）+ `outputs/model_2d.pth`。

### 评估和可视化

```bash
python scripts/result.py --checkpoint weight/best_model.pt --outdir outputs
```

输出相对 L2 误差与三联热力图（预测 / 精确 / 绝对误差）：

```text
./outputs/result_metrics.npz
./outputs/field_result.png
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 原始论文：[A practical PINN framework for multi-scale problems with multi-magnitude loss terms](https://arxiv.org/abs/2308.06672)。
- 原始代码：<https://github.com/wangyong1301108/MMPINN>。
- 本仓库保留来源说明，公开分发前请根据上游项目确认许可证要求。

