<p align="center">
  <strong>
    <span style="font-size: 30px;">ShockCast</span>
  </strong>
</p>

# 模型介绍

ShockCast 是基于论文 A Two-Phase Deep Learning Framework for Adaptive Time-Stepping in High-Speed Flow Modeling 构建的高速流场自适应时间步预测模型。

论文：[A Two-Phase Deep Learning Framework for Adaptive Time-Stepping in High-Speed Flow Modeling](https://arxiv.org/pdf/2506.07969)

本次复现按照论文 Appendix C.3 中 Circular Blast Neural CFL 的设置训练 `800 epoch`，`batch_size=320`，训练噪声 level 为 `0.01`。实际完成 `800/800 epoch`。

# 模型描述
ShockCast-CircularBlast-CFL 使用卷积神经网络从当前 Circular Blast 流场状态中预测下一步时间间隔 delta_t。模型输入为 `xVel`、`yVel`、`density`、`temperature` 四个流场通道，空间分辨率为 `128 x 128`，输出为当前状态对应的下一步时间间隔。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 高速流场时间步预测 | 根据当前流场状态预测自适应时间步 delta_t |
| 神经 PDE 求解器辅助模块 | 可作为 ShockCast 两阶段框架中的 Neural CFL 模块 |

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
modelscope download --model OneScience/ShockCast-CircularBlast-CFL --local_dir ./ShockCast-CircularBlast-CFL
cd ShockCast-CircularBlast-CFL
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
OneScience 社区在 魔搭上 提供 OneScience/shockcast 数据集，可通过以下命令下载。
```text
modelscope download --dataset OneScience/lid_driven_cavity --local_dir ./data
```
本案例使用其中的 `Circular-Blast` 子目录二维圆形爆炸流场数据。数据目录包含 `plt_2.h5` 至 `plt_101.h5` 共 100 个 HDF5 文件，总大小约 1.7 GB。每个文件包含 `density`、`pressure`、`temperature`、`Mach_Number`、`xVel`、`yVel`、`time`、`ratio`、`dir` 等字段，主要流场字段形状为 `(28~52, 128, 128)`，`time` 字段形状为 `(28~52,)`。

### 训练

```bash
python scripts/train.py
```

默认训练会保存 checkpoint：

```text
./weight/best_model.pth
```



### 训练权重
本仓库在 `weight/` 文件夹内提供本次训练得到的模型权重，可用于直接推理。

```text
weight/best_model.pth
```

### 推理

```bash
python scripts/inference.py
```

### 评估和可视化

```bash
python scripts/result.py
```

运行后会在 `results/` 目录下生成推理指标、训练曲线、预测散点图和时间步误差曲线。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- ShockCast 原始论文：[A Two-Phase Deep Learning Framework for Adaptive Time-Stepping in High-Speed Flow Modeling](https://arxiv.org/pdf/2506.07969)。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

