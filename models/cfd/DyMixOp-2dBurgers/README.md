<p align="center">
  <strong>
    <span style="font-size: 30px;">DyMixOp-2dBurgers</span>
  </strong>
</p>

# 模型介绍

DyMixOp-2dBurgers 是基于 DyMixOp 论文构建的二维 Burgers 方程流场预测模型，用于学习速度场随时间推进的代理求解过程。

论文：[DyMixOp: Dynamic Mixture of Operators for Learning Across Heterogeneous PDEs](https://arxiv.org/pdf/2508.13490)

# 模型描述
DyMixOp-2dBurgers 采用局部-全局混合算子和时间尺度自适应动力学层，对 2D Burgers 数据中的 `u/v` 双通道速度场进行多步滚动预测。输入为前 10 帧速度场，输出为后 10 帧速度场，空间分辨率为 `64 x 64`。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 流体方程代理求解 | 面向二维 Burgers 方程的速度场时序预测 |
| 神经算子复现实验 | 用于 DyMixOp 局部-全局混合算子结构的训练、推理和可视化验证 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/DyMixOp-2dBurgers --local_dir ./DyMixOp-2dBurgers
cd DyMixOp-2dBurgers
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
OneScience 社区在 魔搭上 提供 OneScience/DyMixOp-Benchmarks 数据集，可通过以下命令下载：

```text
modelscope download --dataset OneScience/DyMixOp-Benchmarks --include "2dBurgers_1200x20x2x64x64_dt0.0025_t[0_0.5]_nu0.005.mat" --local_dir ./data
```

数据变量为 `uv`，shape 为 `(1200, 21, 2, 64, 64)`，其中前 1000 条轨迹用于训练，后 200 条轨迹用于测试。

### 训练

```bash
python scripts/train.py
```

默认训练会保存 checkpoint：

```text
./weight/best_model.pth
```

### 训练权重
本模型包在 `weight/` 文件夹内提供本次训练得到的模型权重，可用于直接推理。

```text
weight/best_model.pth
```

### 推理

```bash
python scripts/inference.py
```

本次推理使用 200 条测试轨迹，得到 normalized relative MSE 为 `1.0623358757584356e-05`，physical relative MSE 为 `2.31623665895313e-04`，RMSE 为 `0.007739236151728116`，MAE 为 `0.005377683386206627`。论文中 Large 配置的参考 relative MSE 为 `9.18e-04`，本结果为当前 OneScience 生成代码和本次运行设置下的实测结果。

### 评估和可视化

```bash
python scripts/result.py
```

可视化结果保存在 `results/` 目录，包括：

```text
results/final_time_first_sample.png
results/final_time_last_sample.png
results/training_curves.png
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- DyMixOp 原始论文：[DyMixOp: Dynamic Mixture of Operators for Learning Across Heterogeneous PDEs](https://arxiv.org/pdf/2508.13490)。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

