<p align="center">
  <strong>
    <span style="font-size: 30px;">FNO-DeepONet-WaveEq-2605.12997</span>
  </strong>
</p>

# 模型介绍

本模型包复现论文《Frequency Bias and OOD Generalization in Neural Operators under a Variable-Coefficient Wave Equation》（arXiv:2605.12997）。训练 FNO（Fourier Neural Operator）与 DeepONet 两种神经算子架构，学习从初始位移场 u0(x) 与空间变化波速系数 c(x) 到固定终端时刻波动解 u(x,T) 的算子映射，并评估其在频移（OOD-frequency）与系数平滑度移（OOD-smoothness）两种结构化分布偏移下的泛化能力。

论文：Frequency Bias and OOD Generalization in Neural Operators under a Variable-Coefficient Wave Equation
https://arxiv.org/abs/2605.12997

# 模型描述

- **FNO**：Fourier Neural Operator，采用 lifting → 4 层 Fourier layer（谱卷积 + 点态线性）→ projection 结构，retained modes=16，hidden width=64，激活 GELU，配 Dirichlet output envelope 强制边界零值。
- **DeepONet**：branch-trunk 分解结构，branch net 编码输入函数（u0、c），trunk net 编码空间坐标，输出为系数与基函数的内积；branch/trunk hidden=128，latent=128，depth=3，激活 ReLU，含 output bias 与 Dirichlet output envelope。
- 两者均在 1D 变系数波动方程 u_tt = ∂_x(c(x)²u_x)（齐次 Dirichlet 边界）下，用相对 L2 损失训练。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 算子学习训练 | 使用有限差分求解器生成的数据训练 FNO / DeepONet |
| 模型推理 | 加载权重，给定 u0(x) 与 c(x) 预测终端解 u(x,T) |
| OOD 泛化评估 | 在 ID / OOD-frequency / OOD-smoothness 上计算相对 L2 误差 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本 |

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
modelscope download --model OneScience/FNO-DeepONet-WaveEq-2605.12997 --local_dir ./model
cd model
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域
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

训练数据由代码内置的保守格式有限差分求解器实时生成（无外部数据集）。求解器解 u_tt = ∂_x(c(x)²u_x)，网格 128 点，CFL 条件保证数值稳定性，通过随机 Fourier 组合采样初始位移场与波速系数场，并按独立随机种子划分 train/validation/ID/OOD-frequency/OOD-smoothness。

```bash
python scripts/train.py --config conf/params.yaml
```

### 训练

单卡：

```bash
python scripts/train.py --config conf/params.yaml --model-type fno --epochs 30 --seed 2026
python scripts/train.py --config conf/params.yaml --model-type deeponet --epochs 30 --seed 2026
```

训练会在 `outputs/checkpoints/` 下保存 best 与 final 权重（fno_waveeq.pt / deeponet_waveeq.pt）。

### 训练权重

- `weight/fno_waveeq.pt`：FNO best（by validation）模型权重
- `weight/deeponet_waveeq.pt`：DeepONet best 模型权重

### 推理

```bash
python scripts/evaluate.py --config conf/params.yaml --model-type fno --checkpoint weight/fno_waveeq.pt --report-path outputs/eval_fno.json
python scripts/evaluate.py --config conf/params.yaml --model-type deeponet --checkpoint weight/deeponet_waveeq.pt --report-path outputs/eval_deeponet.json
```

### 评估和可视化

评估输出在 ID / OOD-frequency / OOD-smoothness 三个 split 上的相对 L2 误差（并附能量诊断与逐 Fourier 模态谱误差）。

复现结果（Tier1 小数据，30 epochs）：

| 模型 | ID rel-L2 | OOD-frequency rel-L2 | OOD-smoothness rel-L2 |
| --- | --- | --- | --- |
| FNO | 0.246 | 23.92 | 0.253 |
| DeepONet | 0.576 | 0.939 | 0.619 |

定性结论与论文一致：FNO 分布内误差更低，但在频移下显著退化；DeepONet 总体误差较高但退化更平缓；两者在系数平滑度移下均保持稳定。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为《Frequency Bias and OOD Generalization in Neural Operators under a Variable-Coefficient Wave Equation》（arXiv:2605.12997）的复现版本。

