<p align="center">
  <strong>
    <span style="font-size: 30px;">FPINNs</span>
  </strong>
</p>

# 模型介绍

本模型包中的 FPINNs 指 Fuzzy Physics-Informed Neural Networks，而非分数阶 PINNs。模型在全连接网络旁增加高斯模糊隶属度分支，将神经特征和模糊规则特征融合后预测偏微分方程解。

当前案例求解 Allen-Cahn 方程：

```text
u_t - lambda_1 u_xx + lambda_2 (u^3 - u) = 0
```

论文：Deep fuzzy physics-informed neural networks for forward and inverse PDE problems  
https://doi.org/10.1016/j.neunet.2024.106750

# 模型描述

FPINNs 联合数据损失和 PDE 残差进行训练，支持 Allen-Cahn 方程的正向和逆向任务。正向任务在已知 `lambda_1=0.0001`、`lambda_2=5.0` 时预测时空解；逆向任务则根据观测数据同时学习方程解及这两个参数。模型默认使用 Adam 训练，并可选择 L-BFGS 精调。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| Allen-Cahn 正向求解 | 使用已知扩散和反应参数预测完整时空解。 |
| Allen-Cahn 逆向求解 | 从解观测中识别扩散和反应参数。 |
| 模糊特征研究 | 比较神经特征和高斯模糊规则特征的融合效果。 |
| 模型流程验证 | 使用随包数据和小规模配置检查训练、推理流程。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 进行训练和完整网格推理。
- CPU 可用于小配置连通性验证。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

### 下载模型包

```bash
modelscope download --model OneScience/FPINNs --local_dir ./FPINNs
cd FPINNs
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

模型包内置 Allen-Cahn 数据文件 `data/AC.mat`，其中包含空间坐标、时间坐标和对应的方程解。可在 `conf/config.yaml` 中调整训练采样点数量和评估批大小。

### 训练

训练任务由 `conf/config.yaml` 中的 `common.task` 控制：`forward` 为正向任务，`inverse` 为逆向任务。配置完成后运行：

```bash
python scripts/train.py
```

训练检查点和历史记录默认保存至 `weight/` 和 `result/` 目录。

### 训练权重

本仓库在`weight/`文件夹内提供基于Allen-Cahn 数据训练的权重。

### 推理、评估和可视化

完成对应任务训练后运行：

```bash
python scripts/inference.py
```

推理任务同样由 `common.task` 控制，结果默认保存为 `result/fpinn_forward.*` 或 `result/fpinn_inverse.*`。逆向任务还会输出恢复得到的 `lambda_1` 和 `lambda_2`。模型、训练、损失和推理参数均可在 `conf/config.yaml` 中修改。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Wu, W., Duan, S., Sun, Y., Yu, Y., Liu, D., and Peng, D. Deep fuzzy physics-informed neural networks for forward and inverse PDE problems. Neural Networks, 181, 106750, 2025.
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
