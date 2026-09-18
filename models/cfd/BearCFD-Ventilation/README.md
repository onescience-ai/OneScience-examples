<p align="center">
  <strong>
    <span style="font-size: 30px;">BearCFD-Ventilation</span>
  </strong>
</p>

# 模型介绍

BearCFD-Ventilation 是基于 BEAR-CFD 数据集构建的室内通风 CO2 浓度预测模型，可根据历史 CO2 分布和通风控制参数预测未来 CO2 浓度变化。

论文：[Building Control CFD: Efficient Deep Learning of Indoor CO2 Dynamics from Sustainable CFD Simulations](https://arxiv.org/pdf/2504.21243)

本仓库由 OneScience 技能流程依据论文描述与官方配置，独立复现 BearCFD-Ventilation 在 Bear-CFD-dataset 数据集上的实验。

# 模型描述
BearCFD-Ventilation 基于神经算子 Transformer 结构，适配 BEAR-CFD 室内通风瞬态数据训练，面向人员区域 CO2 浓度开展多步预测。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 室内通风预测 | 根据送风速度、送风角度和人员数量预测室内 CO2 浓度 |
| CFD 仿真加速 | 面向室内通风瞬态 CFD 结果构建快速代理模型 |

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
modelscope download --model OneScience/BearCFD-Ventilation --local_dir ./BearCFD-Ventilation
cd BearCFD-Ventilation
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
OneScience 社区在 魔搭上 提供 OneScience/Bear-CFD-dataset 数据集，可通过以下命令下载：

```text
modelscope download --dataset OneScience/Bear-CFD-dataset --local_dir ./data
```

该目录包含 `unsteady_10.pkl` 至 `unsteady_41.pkl`。每个样本包含人员区域 CO2 浓度、通风入口速度、通风入口角度和人员数量等信息，训练前请确认 `config/config.yaml` 中数据路径设置正确。

### 训练

```bash
python scripts/train.py
```

默认训练会保存 checkpoint：

```text
./weight/best_model.pth
```

### 训练权重
本仓库在 `weight/` 文件夹内提供基于 BEAR-CFD 数据训练得到的模型权重，可用于直接推理。

```text
weight/best_model.pth
```

本次测试集结果为 `relative_l2=0.144541`，`rmse=94.616395`。论文 Table 3 中 ensemble 测试 `l2 error` 为 10.90%，本次结果对应 32 个官方原始样本上的运行记录。

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

- 原始论文：[Building Control CFD: Efficient Deep Learning of Indoor CO2 Dynamics from Sustainable CFD Simulations](https://arxiv.org/pdf/2504.21243)。
- 数据集：[BEAR-CFD dataset](https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset)。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

