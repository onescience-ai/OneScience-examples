<p align="center">
  <strong>
    <span style="font-size: 30px;">GFNO</span>
  </strong>
</p>

# 模型介绍

GFNO（Group Equivariant Fourier Neural Operator）是一种引入群等变结构的 Fourier 神经算子，用于学习具有几何对称性的物理系统。本模型面向二维规则网格上的 Navier-Stokes 时序预测，默认使用 C4 旋转群、前 10 个时间步预测后续 10 个时间步。

论文：Group Equivariant Fourier Neural Operators for Partial Differential Equations  
https://openreview.net/forum?id=kgAOY5x4fi

# 模型描述

GFNO 在 Fourier Neural Operator 的谱卷积基础上引入群等变权重共享，使网络在旋转或反射变换下保持一致响应，并更充分地利用 Navier-Stokes 流场中的几何对称性。模型支持 C4 旋转群和 D4 二面体群，通过自回归方式完成多步流场预测。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 流场时序预测 | 根据历史涡量场预测未来 Navier-Stokes 状态。 |
| 几何等变研究 | 比较普通 FNO 与 C4/D4 群等变算子的表现。 |
| CFD 快速代理 | 学习规则网格上历史物理场到未来物理场的映射。 |
| 模型流程验证 | 使用小规模配置检查模型训练、推理和结果可视化流程。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 进行训练和推理。
- CPU 可用于小配置连通性验证，完整训练速度较慢。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

### 下载模型包

```bash
modelscope download --model OneScience/GFNO --local_dir ./GFNO
cd GFNO
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

模型使用 Navier-Stokes 标准数据集 `NavierStokes_V1e-5_N1200_T20.mat`，数据变量 `u` 的形状为 `[1200, 64, 64, 20]`。可通过以下命令下载数据，并确认 `conf/config.yaml` 中的数据路径配置正确：

```bash
modelscope download --dataset OneScience/cfd_benchmark data/ns/NavierStokes_V1e-5_N1200_T20.mat --local_dir ./data
```

### 训练

```bash
python scripts/train.py
```

默认权重保存至 `weight/gfno_navier_stokes.pt`。

### 训练权重

本仓库在`weight/`文件夹内提供基于Navier-Stokes 标准数据集训练的权重。

### 推理、评估和可视化

完成训练后运行：

```bash
python scripts/inference.py
```

脚本默认读取 `weight/gfno_navier_stokes.pt`，预测张量和可视化结果保存在 `result/` 目录。训练和推理参数均可在 `conf/config.yaml` 中修改。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Romero, D. W., Bekkers, E. J., Tomczak, J. M., and Hoogendoorn, M. Group Equivariant Fourier Neural Operators for Partial Differential Equations.
- Li, Z. et al. Fourier Neural Operator for Parametric Partial Differential Equations. arXiv:2010.08895, 2020.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
