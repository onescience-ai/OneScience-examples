<p align="center">
  <strong>
    <span style="font-size: 30px;">HardBCPINN-SinePipeFlow</span>
  </strong>
</p>

# 模型介绍

HardBCPINN-SinePipeFlow 是基于论文中的硬边界条件 PINN 方法构建的二维不可压缩稳态管流复现模型，用于求解带正弦边界的管道内流场。

论文：[Solving Navier-Stokes Equations Using Data-free Physics-Informed Neural Networks With Hard Boundary Conditions](https://arxiv.org/pdf/2511.14497)

# 模型描述
HardBCPINN-SinePipeFlow 采用数据无关的物理约束训练方式，模型输入二维坐标 `(x, y)`，输出速度和压力 `(u, v, p)`。实现中将入口压力、出口压力和壁面无滑移边界条件写入输出变换，使模型在训练过程中直接满足硬边界条件，并通过 Navier-Stokes 方程残差优化内部流场。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 管道内流场求解 | 求解带正弦边界的二维稳态管流速度场和压力场 |
| PINN 方法验证 | 验证硬边界条件约束下的数据无关 PINN 训练流程 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/HardBCPINN-SinePipeFlow --local_dir ./HardBCPINN-SinePipeFlow
cd HardBCPINN-SinePipeFlow
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
本案例复现的是论文第 3.3 节的 Re=100 正弦边界管流实验，属于 data-free PINN 任务，不需要外部标注数据集。训练点、边界点和可视化网格由脚本根据论文中的几何与边界条件生成：

```bash
python scripts/fake_data.py
```

默认几何参数为 `R0=0.05`、`A=0.005`、`N=6`、`L=1.0`，入口压力为 `0.1`，出口压力为 `0.0`。

### 训练

```bash
python scripts/train.py
```

默认训练会保存 checkpoint：

```text
./weight/best_model.pth
```

### 训练权重
本仓库在 `weight/best_model.pth` 文件夹内提供已训练得到的模型权重，可用于直接推理。


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

- 原始论文：[Solving Navier-Stokes Equations Using Data-free Physics-Informed Neural Networks With Hard Boundary Conditions](https://arxiv.org/pdf/2511.14497)。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。
