<p align="center">
  <strong>
    <span style="font-size: 30px;">MTL-Airfoil-Surrogate</span>
  </strong>
</p>

# 模型介绍

MTL-Airfoil-Surrogate 是基于多任务学习（Multi-Task Learning, MTL）的二维翼型流场代理模型，
将翼型**表面**预测（表面压力分布）与**流场域**预测（速度场、压力场、湍流涡黏）分离，
使用双解码器（Decoder I）并配合 STCH（Smooth Tchebycheff Scalarization）多任务损失平衡，
可对 AirfRANS 非结构网格翼型数据开展气动性能快速预测。

论文：[Enhancing Airfoil Design Optimization Surrogate Models Using Multi-Task Learning: Separating Airfoil Surface and Fluid Domain Predictions](https://doi.org/10.1063/5.0258928) (Physics of Fluids, 2025)

# 模型描述

MTL-Airfoil-Surrogate 是基于 MLP 主干与 Decoder I 的多任务代理建模方法，以 AirfRANS 非结构网格节点的二维坐标、自由来流速度、翼型符号距离函数和表面法向量构成 7 维输入，经 `7→64→64→8` 编码器及带批归一化的 `8→64→64→64→8` 共享主干提取气动特征，再由两个独立解码器分别预测翼型表面压力 $p|_S$ 与流体域速度、压力及湍流运动黏度 $(u,v,p,\nu_t)|_V$；训练采用 STCH 平滑切比雪夫标量化动态平衡表面与流场域的均方误差，从而缓解任务间的优化冲突，并提升流场及升阻力等气动性能指标的预测精度。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 翼型气动设计 | 快速预测二维翼型表面压力与流场，用于候选外形筛选 |
| 气动优化代理 | 升力/阻力系数的代理求解，替代高成本 CFD 仿真 |

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
modelscope download --model OneScience/MTL-Airfoil-Surrogate --local_dir ./MTL-Airfoil-Surrogate
cd MTL-Airfoil-Surrogate
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

### 训练数据介绍

OneScience 社区提供可供训练的 `AirfRANS`，可通过下述命令下载，并确认 `config/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/airfrans --local_dir ./data
```

### 训练

按照论文模型训练配置设置的参数：Adam lr=1e-3，OneCycleLR max 1e-3，800 epochs，batch_size=1，
32000 点子采样，seed=1，STCH mu=1 warmup 4 epochs。

```bash
python scripts/train.py --data_dir ./data --out outputs
```

默认训练会保存 checkpoint：

```text
./outputs/best_model.pt
./outputs/metrics.json
```

### 训练权重

本仓库在 `weight/` 文件夹内提供基于 AirfRANS 数据预训练的模型权重，可用于直接推理。

### 推理

```bash
python scripts/inference.py --checkpoint weight/best_model.pt \
    --data_dir ./data --stats_dir ./data/stats --out outputs/predictions.pt
```

### 评估和可视化

```bash
python scripts/result.py --checkpoint weight/best_model.pt \
    --data_dir ./data --stats_dir ./data/stats --out outputs/metrics.json
```

评估指标：

| 指标 | 说明 |
| :--- | :--- |
| `relative_error_force_x` | 升力系数相对误差 |
| `relative_error_force_y` | 阻力系数相对误差 |
| `spearman_coef_y` | 阻力系数 Spearman 秩相关 |
| `surface_p` / `volume_*` | 归一化场的逐场 MSE |

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 原始论文：[Enhancing Airfoil Design Optimization Surrogate Models Using Multi-Task Learning: Separating Airfoil Surface and Fluid Domain Predictions](https://doi.org/10.1063/5.0258928)。
- 原始代码：<https://github.com/Gxinhu/MTL_airfoil_surrogate>。
- 本仓库保留来源说明，公开分发前请根据上游项目确认许可证要求。

