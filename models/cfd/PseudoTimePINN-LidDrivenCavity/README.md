<p align="center">
  <strong>
    <span style="font-size: 30px;">PseudoTimePINN-LidDrivenCavity</span>
  </strong>
</p>

# 模型介绍

PseudoTimePINN-LidDrivenCavity 是基于论文 When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious Solutions 构建的二维顶盖驱动腔流复现模型，用于求解 Re=5000 条件下的稳态不可压缩 Navier-Stokes 方程。

论文：[When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious Solutions](https://arxiv.org/pdf/2604.23528)

# 模型描述
PseudoTimePINN-LidDrivenCavity 采用 PINN 形式从空间坐标预测速度和压力场，输入为二维坐标 `(x, y)`，输出为 `(u, v, p)`。模型中包含随机 Fourier 特征、PirateNet 风格的多层网络、边界条件损失、Navier-Stokes 残差损失以及 pseudo-time residual 项。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 腔体流动预测 | 复现 Re=5000 顶盖驱动腔流的速度场和压力场 |
| PINN 数值求解 | 用物理约束神经网络求解二维稳态不可压缩流动 |

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
modelscope download --model OneScience/PseudoTimePINN-LidDrivenCavity --local_dir ./PseudoTimePINN-LidDrivenCavity
cd PseudoTimePINN-LidDrivenCavity
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
OneScience 社区在 魔搭上 提供 OneScience/lid_driven_cavity 数据集，可通过以下命令下载：
```bash
modelscope download --dataset OneScience/lid_driven_cavity --local_dir ./data
```

数据中包含 `x`、`y`、`u`、`v`、`nu` 字段，其中 `x` 和 `y` 为 `1x256` 网格坐标，`u` 和 `v` 为 `256x256` 参考速度场，`nu=0.0002`。确保`config/config.yaml` 中的 `data_dir` 和 `data_file`指向数据下载的实际路径。


### 训练

```bash
python scripts/train.py
```

默认训练会保存 checkpoint：

```text
./weight/best_model.pth
```

### 训练权重
本仓库在 `weight/` 文件夹内提供本次复现训练得到的模型权重，可用于直接推理。


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

- 原始论文：[When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious Solutions](https://arxiv.org/abs/2604.23528)。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

