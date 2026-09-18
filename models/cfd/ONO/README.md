<p align="center">
  <strong>
    <span style="font-size: 30px;">ONO</span>
  </strong>
</p>

# 模型介绍

ONO（Orthogonal Neural Operator）是一种引入正交注意力机制的神经算子，用于学习复杂物理系统的演化规律。本模型面向二维规则网格上的 Navier-Stokes 时序预测，默认使用前 10 个时间步预测后续 10 个时间步。

论文：Improved Operator Learning by Orthogonal Attention  
https://arxiv.org/abs/2310.12487

# 模型描述

ONO 通过注意力机制提取网格点之间的全局关系，并利用协方差白化和正交投影更新物理场特征，以缓解深层神经算子中的特征退化和过平滑问题。模型支持 Nyström、线性和标准自注意力，可通过自回归方式完成多步流场预测。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 流场时序预测 | 根据历史涡量场预测未来 Navier-Stokes 状态。 |
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
modelscope download --model OneScience/ONO --local_dir ./ONO
cd ONO
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

默认权重保存至 `weight/ono_navier_stokes.pt`。

### 训练权重

本仓库在`weight/`文件夹内提供基于Navier-Stokes 标准数据集训练的权重

### 推理、评估和可视化

```bash
python scripts/inference.py
```

脚本默认读取 `weight/ono_navier_stokes.pt`，预测张量和可视化结果保存在 `result/` 目录。训练和推理参数均可在 `conf/config.yaml` 中修改。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Improved Operator Learning by Orthogonal Attention. ICML, 2024.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
