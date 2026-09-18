<p align="center">
  <strong>
    <span style="font-size: 30px;">FactFormer</span>
  </strong>
</p>

# 模型介绍

FactFormer（Factorized Transformer）是一种面向偏微分方程代理建模的 Transformer 模型。本模型用于二维 Kolmogorov Flow 涡量场时序预测，默认使用前 10 个时间步预测后续 16 个时间步。

论文：[Scalable Transformer for PDE Surrogate Modeling](https://arxiv.org/abs/2305.17560)

# 模型描述

FactFormer 将二维规则网格上的全局注意力分解为沿两个空间轴分别计算的注意力，在保留长程空间信息交互能力的同时，降低计算和显存开销。模型通过 latent propagation 分块预测未来状态，并以自回归方式完成完整时间窗口的推演。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 涡量场时序预测 | 根据历史二维涡量场预测未来 Kolmogorov Flow 状态。 |
| PDE 代理建模 | 学习规则网格上历史物理场到未来物理场的映射。 |
| 因子化注意力研究 | 验证轴向注意力在高分辨率物理场建模中的效果。 |
| 模型流程验证 | 使用随包权重或小规模配置检查训练和推理流程。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 进行训练和推理。
- CPU 可用于小配置连通性验证，默认配置训练速度较慢。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

### 下载模型包

```bash
modelscope download --model OneScience/FactFormer --local_dir ./FactFormer
cd FactFormer
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

模型使用二维 Kolmogorov Flow 数据，原始数组形状为 `[120, 320, 256, 256]`，依次表示轨迹、时间步和两个空间维度。数据文件较大，程序会通过内存映射按需读取。

可通过以下命令下载数据，并确认 `conf/config.yaml` 中的数据路径配置正确：

```bash
modelscope download --dataset OneScience/Kolmogorov_flow_2d --local_dir ./data
```

### 训练

```bash
python scripts/train.py
```

默认权重保存至 `weight/factformer_kolmogorov.pt`。

### 训练权重

本仓库在`weight/`文件夹内提供基于二维 Kolmogorov Flow 数据集训练的权重。

### 推理、评估和可视化

模型包提供用于流程验证的权重，可在准备数据后直接运行：

```bash
python scripts/inference.py
```

脚本默认读取 `weight/factformer_kolmogorov.pt`，预测张量和可视化结果保存在 `result/` 目录。训练和推理参数均可在 `conf/config.yaml` 中修改。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Li, Shu, and Barati Farimani. [Scalable Transformer for PDE Surrogate Modeling](https://arxiv.org/abs/2305.17560).
- Kolmogorov Flow 数据用于 Re=1000 二维涡量场预测任务。
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
