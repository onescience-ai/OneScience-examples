<p align="center">
  <strong>
    <span style="font-size: 30px;">PINTO-Kovasznay</span>
  </strong>
</p>

# 模型介绍

PINTO-Kovasznay 是基于印度科学理工学院（Indian Institute of Science，IISc）计算与数据科学系 QUEST 实验室提出的物理信息 Transformer 神经算子 PINTO 构建的二维稳态流场代理模型，可融合空间坐标、边界条件与 Navier–Stokes 方程约束，快速预测 Kovasznay 流动的速度场和压力场

论文：[PINTO: Physics-informed transformer neural operator for learning generalized solutions of partial differential equations for any initial and boundary condition](https://arxiv.org/abs/2412.09009)

# 模型描述

PINTO-Kovasznay 以待预测点坐标、边界点坐标及边界物理量 [u, v, p] 为输入，通过查询映射、边界位置编码、边界值编码和多头交叉注意力融合边界条件，最终输出二维速度分量与压力场。训练过程结合解析解数据误差、边界条件误差以及稳态不可压缩 Navier–Stokes 方程残差。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| Kovasznay 流场预测 | 预测不同运动黏度条件下 Kovasznay 稳态流动的速度场和压力场 |
| 稳态 Navier–Stokes 求解 | 近似求解二维不可压缩稳态 Navier–Stokes 方程 |
| 边界条件感知建模 | 利用交叉注意力融合边界位置及边界物理量，学习边界条件到流场解的映射|

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入、前向和小配置连通性验证。
- DCU 用户需要预先安装 DTK，并使用与当前集群匹配的 OneScience Python 环境。

### 下载模型包

```bash
modelscope download --model OneScience/PINTO-Kovasznay --local_dir ./PINTO-Kovasznay
cd PINTO-Kovasznay
```

### 安装运行环境

**DCU 环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

本模型不需要下载外部数据集。`scripts/fake_data.py` 会根据 Kovasznay 解析解自动生成训练样本、边界样本和测试样本。配置见`config/config.yaml`
默认训练数据包括：
- 计算区域：\(x\in[-0.5,1.0]\)，\(y\in[-0.5,1.5]\)
- 训练黏度 \(\nu\)：0.05、1/30、0.02、0.0125
- 内部查询点：拉丁超立方采样
- 边界点：从矩形区域的上、下、左、右四条边随机采样
- 输入：查询点坐标、边界点坐标及边界上的 [u,v,p]
- 标签：Kovasznay 解析解计算得到的速度分量 \(u,v\) 和压力 \(p\)
当前默认训练脚本运行的是最小冒烟训练：每个黏度取 32 个内部点，每条边取 5 个点，共 20 个边界点，并只执行一次参数更新。因此，它更接近解析流场复现实验数据，而不是一个预先下载并存储的标准数据集。



### 训练

```bash
python scripts/train.py
```

默认训练会保存 checkpoint：

```text
weight/best_model.pt
```

### 训练权重

本模型包在 `weight/` 目录下提供一次标准配置运行得到的权重：

```text
weight/best_model.pt
```


### 推理

```bash
python scripts/inference.py
```

推理结果默认保存到：

```text
results/metrics.json
results/predictions.npz
```

当前权重对应的推理指标：

| 指标 | 数值 |
| :--- | ---: |
| relative_l2_u | 1.029836654663086 |
| relative_l2_v | 1.3754936456680298 |
| relative_l2_p | 1.0770763158798218 |
| relative_l2_mean | 1.1608022054036458 |

### 评估和可视化

```bash
python scripts/result.py
```

结果汇总文件默认保存到：

```text
results/summary.json
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- PINTO 原始论文：[PINTO: Physics-informed transformer neural operator for learning generalized solutions of partial differential equations for any initial and boundary condition](https://arxiv.org/abs/2412.09009)。
- 本模型包基于论文复现实验整理，保留来源说明，并面向 OneScience 模型运行和论文复现评测场景使用。

