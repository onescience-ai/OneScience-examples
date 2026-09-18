<p align="center">
  <strong>
    <span style="font-size: 30px;">Transolver-Car-Design</span>
  </strong>
</p>

# 模型介绍

Transolver-Car-Design 是基于清华大学 THUML 团队提出的 Transolver / Transolver++ 构建的三维汽车外流场预测模型，可用汽车流场代理建模和阻力系数快速预测。

论文：[Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/pdf/2402.02366)

# 模型描述
Transolver-Car-Design 基于引入 Physics-Attention 的 Transformer 架构，使用 ShapeNet-Car 汽车气动仿真数据进行训练，面向复杂汽车几何开展速度场、压力场及阻力系数预测。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 汽车气动设计 | 快速预测车辆外流场速度和表面压力 |
| CFD 代理建模 | 用神经网络近似复杂非结构网格上的流体求解过程 |
| 仿真加速 | 为大规模候选设计筛选提供轻量评估链路 |

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
modelscope download --model OneScience/Transolver-Car-Design --local_dir ./Transolver-Car-Design
cd Transolver-Car-Design
```

### 安装运行环境


**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

OneScience 社区提供可供训练的 `ShapeNetCar` 数据，用户可通过下述命令下载，并确认 `config/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/ShapeNetCar --local_dir ./data
```

### 训练

```bash
python scripts/train.py
```

训练会在weight下保存 Transolver_plus.pth

```text
./weight/Transolver_plus.pth
```
### 训练权重
本仓库在weights/文件夹内提供基于ShapeNetCar数据预训练的模型权重， 该权重即将上传。

### 推理

```bash
python scripts/inference.py
```

### 评估和可视化

```bash
python scripts/result.py
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Transolver 原始论文：[Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/pdf/2402.02366)。
- Transolver++ 原始论文：[Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries](https://arxiv.org/abs/2502.02414)
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。
