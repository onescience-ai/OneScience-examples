<p align="center">
  <strong>
    <span style="font-size: 30px;">BCAT-NS2D</span>
  </strong>
</p>

# 模型介绍

BCAT-NS2D 是基于加利福尼亚大学洛杉矶分校（UCLA）数学系与卡内基梅隆大学（CMU）数学科学系联合提出的块因果 Transformer 模型 BCAT 构建的二维流体序列预测模型，可利用历史流场作为上下文，自回归预测后续时刻的速度与涡量演化。

论文：[BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics](https://arxiv.org/abs/2501.18972)

# 模型描述

BCAT-NS2D 以包含速度分量和涡量 [u, v, omega] 的二维流场序列为输入，将每个时间步的流场划分为多个空间 patch，并融合时间嵌入与 patch 位置嵌入。模型通过块因果注意力掩码限制未来时间块的信息泄漏，利用 Transformer 编码历史时空特征，最终预测后续时间步的速度场与涡量场。当前实现采用周期边界、散度自由的合成二维涡流序列，并通过均方误差进行训练。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 二维流场序列预测 | 根据历史流场连续预测后续时刻的速度分量和涡量分布 |
| Navier–Stokes Rollout | 近似模拟二维不可压缩 Navier–Stokes 流动的时序演化 |
| Transformer PDE 模型验证 | 验证流场 patch 编码、时空嵌入及块因果注意力机制的预测流程 |

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
modelscope download --model OneScience/BCAT-NS2D --local_dir ./BCAT-NS2D
cd BCAT-NS2D
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

本模型不需要下载外部数据集。`scripts/fake_data.py` 会自动生成周期边界、散度自由的二维涡流序列，作为训练和推理样本。

配置文件为`config/config.yaml`，默认数据配置如下：
- 网格大小：\(16\times16\)
- 序列长度：5 个时间步
- 物理通道：水平速度 \(u\)、垂直速度 \(v\) 和涡量 \(\omega\)
- 运动黏度：\(0.001\)
- 样本变化：随机设置涡流初始相位及水平、垂直移动速度
- 时间演化：流场随平移和黏性作用逐渐衰减
- 训练目标：根据当前及历史流场预测下一时间步的 [u,v,\omega]
该数据主要用于验证 BCAT 的二维流场时序预测与 rollout 流程


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
| relative_l2 | 1.2696276903152466 |
| mse | 0.713721513748169 |

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

- BCAT 原始论文：[BCAT: A Block Causal Transformer for PDE Foundation Models for Fluid Dynamics](https://arxiv.org/abs/2501.18972)。
- 本模型包基于论文复现实验整理，保留来源说明，并面向 OneScience 模型运行和论文复现评测场景使用。

