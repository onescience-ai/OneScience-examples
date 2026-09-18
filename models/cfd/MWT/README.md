<p align="center">
  <strong>
    <span style="font-size: 30px;">MWT</span>
  </strong>
</p>

# 模型介绍
MWT（Multiwavelet-based Operator Learning）是由 Gaurav Gupta、Xiongye Xiao 和 Paul Bogdan 提出的多小波算子学习框架。该模型利用正交多项式构造固定的多小波分解与重构滤波器，在多尺度空间中学习微分方程解算子，从而实现具有数据效率和分辨率泛化能力的物理场预测。

本仓库由 OneScience 技能流程依据论文描述，独立复现 MWT 的二维 Navier–Stokes 涡量预测实验。模型在周期性单位环面上，将前 10 个时间帧的涡量场与空间、时间坐标组合为输入，一次性预测其余 \(T-10\) 个时间帧的涡量场。实验使用由 \(256\times256\) 下采样得到的 \(64\times64\) 规则网格数据。

论文：[Multiwavelet-based Operator Learning for Differential Equations](https://arxiv.org/abs/2109.13459)

# 模型描述
MWT 是面向微分方程算子学习的多尺度神经网络架构。模型采用“输入升维—多小波分解—多尺度算子映射—多小波重构—涡量投影”结构：首先通过线性层将 13 维输入升维至 \(c k^2=36\) 维特征，其中 \(c=4\)、Legendre 多小波阶数 \(k=3\)；随后使用论文给定的固定 Legendre 滤波矩阵及其二维 Kronecker 积，在两个空间维度上递归执行多小波分解。每个尺度上的细节系数和光滑系数分别通过可学习的 \(A\)、\(B\)、\(C\) 算子映射，其中当前实现采用作用于空间—时间维度 \((x,y,t)\) 的三维 Fourier 谱卷积与逐点卷积。最粗尺度由 \(\bar{T}\) 映射处理，再通过固定滤波器逐级重构至原始分辨率。

针对二维 Navier–Stokes 实验，模型堆叠 4 个 MWT Block，并在 Block 之间使用 BatchNorm3d 和 ReLU 非线性。重构后的 36 维特征依次通过 \(36\rightarrow128\rightarrow1\) 的输出头，得到每个空间网格点和预测时刻对应的涡量值。


## 适用场景

| 场景 | 说明 |
| --- | --- |
| Navier–Stokes 涡量预测 | 根据前 10 个时间帧的二维涡量场，一次性预测后续时间范围内的涡量演化 |
| 规则网格算子学习 | 对周期性规则网格上的输入函数与输出函数之间的映射进行建模 |
| 多尺度物理场建模 | 利用固定多小波分解和可学习的尺度内算子捕获不同空间尺度上的流场特征 |
| 时空场快速推理 | 在训练数据分布和黏性系数条件下近似数值求解器，实现完整预测时间区间的快速批量推理 |


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
modelscope download --model OneScience/MWT --local_dir ./MWT
cd MWT
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

OneScience 社区在 魔搭上 提供 [`OneScience/fno`](https://modelscope.cn/datasets/OneScience/fno) 数据集，可通过以下命令下载：

```bash
modelscope download --dataset OneScience/fno --local_dir ./data
```

请确保 config/config.yaml 中的 paths.data_root 指向该目录。当前复现实验使用以下数据文件：
- ns_V1e-3_N5000_T50.mat：黏性系数 \(\nu=10^{-3}\)，包含 5000 个样本和 50 个时间帧；
- ns_V1e-4_N10000_T30.mat：黏性系数 \(\nu=10^{-4}\)，包含 10000 个样本；实验按照配置截取前 30 个时间帧；
- NavierStokes_V1e-5_N1200_T20.mat：黏性系数 \(\nu=10^{-5}\)，包含 1200 个样本和 20 个时间帧。
MAT 文件中的主要变量包括：
- u：Navier–Stokes 涡量随时间的演化结果。数据加载后统一转换为 [num_samples, 64, 64, T]；
- t：时间坐标，数据加载后转换为 [T]；
- a：用于生成数值解的初始条件；当前模型直接使用 u 的前 10 个时间帧，因此不单独读取 a 作为模型输入。

### 训练

默认配置 config/config.yaml 包含论文中的四组 MWT Navier–Stokes 实验。默认激活的实验为 ns_1e-3_t50，可通过 --experiment 选择其他实验：

```bash
python scripts/train.py \
  --config config/config.yaml \
  --experiment ns_1e-3_t50 \
  --seed 0
```
支持的实验名称包括：
- ns_1e-3_t50：\(\nu=10^{-3}\)、\(T=50\)、名义训练样本数 1000、训练 500 epochs；
- ns_1e-4_t30_n1000：\(\nu=10^{-4}\)、\(T=30\)、名义训练样本数 1000、训练 500 epochs；
- ns_1e-4_t30_n10000：\(\nu=10^{-4}\)、\(T=30\)、名义训练样本数 10000、训练 200 epochs；
- ns_1e-5_t20：\(\nu=10^{-5}\)、\(T=20\)、名义训练样本数 1000、训练 500 epochs。

### 训练权重

本仓库在 `weight/` 文件夹内提供基于 Navier–Stokes 数据训练的 MWT 模型权重，可直接加载用于推理和数值评估。

### 推理评估可视化

以下命令在 checkpoint 记录的固定200 个测试样本上执行推理，并在完成后输出物理空间 mean relative L2：

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint weight/best_model.pt
```

### 评估和可视化

运行前请确认 `config/config.yaml` 中的 `data.root` 路径有效，且 `weight/best_model.pth` 已存在。

数值评估由 `scripts/inference.py` 完成。脚本加载最佳 checkpoint，在固定测试集上执行 one-shot 涡量预测，反归一化预测值与目标值，并计算物理空间 relative L2。

```bash
python scripts/inference.py \
  --config config/config.yaml \
  --checkpoint weight/best_model.pt \
  --device auto \
  --batch-size 1 \
  --output-dir results
```

推理结果默认保存至：

```text
results/
├── inference_metrics.json
├── predictions.npy
├── targets.npy
└── sample_indices.npy
```

完成推理后，运行独立可视化脚本：

```bash
python scripts/result.py --config config/config.yaml  --sample 0
```

可视化结果默认保存至：

```text
results/
├── field_comparison.png
├── relative_l2_over_time.png
└── result_summary.json
```

其中：

- field_comparison.png：展示所选测试样本在预测时间轴起始、中间和末尾时刻的真实涡量场、MWT 预测涡量场及绝对误差；真实值与预测值使用一致的发散色标，绝对误差使用非负顺序色标；
- relative_l2_over_time.png：展示整个 one-shot 预测时间范围内，各预测时刻的空间 relative L2 变化；
- result_summary.json：记录实验名称、运行层级、样本数量、数组形状、mean relative L2、MAE、RMSE、最大绝对误差、论文参考值、可比性标记及可视化文件路径。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 论文：[Multiwavelet-based Operator Learning for Differential Equations](https://proceedings.neurips.cc/paper/2021/file/c9e5c2b59d98488fe1070e744041ea0e-Paper.pdf)，NeurIPS 2021；预印本：[arXiv:2109.13459](https://arxiv.org/abs/2109.13459)。
- 官方实现：[gaurav71531/mwt-operator](https://github.com/gaurav71531/mwt-operator)。
- 本仓库为依据论文描述与官方配置完成的独立复现，不代表论文作者或官方实现。论文、官方代码、数据集及其他第三方资源分别受其原始版权声明、许可证和使用条款约束。

