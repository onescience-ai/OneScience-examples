<p align="center">
  <strong>
    <span style="font-size: 30px;">Spherical Fourier Neural Operators</span>
  </strong>
</p>


# 模型介绍

SFNO（Spherical Fourier Neural Operator）由 NVIDIA 与 Caltech 等单位联合提出，将标准 FNO 中的平面 FFT 替换为球谐变换（SHT），使频域卷积尊重球面几何，缓解极区伪影、频谱伪影与长时间自回归滚动的失稳问题。

论文：Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere

https://arxiv.org/abs/2306.03838

# 模型描述

SFNO 是确定性、单状态推进的全球天气动力学模型：输入一个 6 小时时刻的大气状态，输出下一 6 小时时刻的同一变量集合，长时效通过自回归滚动获得。本仓库基于官方 `NVIDIA/torch-harmonics` 中的 reference implementation 整理并接入 OneScience 数据读取与训练流程。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气动力学研究 | 基于 ERA5 数据训练球面等变的神经算子预报模型。 |
| 长时间自回归稳定性研究 | 验证模型在多步滚动中的极区伪影与耗散行为。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、训练入口、推理和结果脚本。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。
- 模型依赖 `torch-harmonics`（球谐变换），安装前请确认其与 PyTorch/CUDA 版本匹配。


### 下载模型包

```bash
modelscope download --model OneScience/SphericalFourierNeuralOperators --local_dir ./SphericalFourierNeuralOperators
cd SphericalFourierNeuralOperators
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[earth-dcu] torch-harmonics -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[earth-gpu] torch-harmonics -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

OneScience 社区提供可供训练的 ERA5 数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认 `conf/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

如需快速验证流程，可先运行虚拟数据脚本：

```bash
python scripts/fake_data.py
```

> 注：`scripts/fake_data.py` 根据模型配置生成 `[T, C, H, W]` HDF5 数据；当前小配置为 6 通道、32×64 网格，并自动计算满足 batch 的时间长度。

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练输出：

```text
data/checkpoints/model_bak.pth
data/checkpoints/trloss.npy
data/checkpoints/valoss.npy
```

### 训练权重
本仓库在 weight/ 文件夹内预留权重目录。论文未明确公开 26/73 通道天气模型权重；本仓库当前不提供官方权重，用户可依据论文配置自行训练。

### 推理

推理默认读取 `data/checkpoints/model_bak.pth`：

```bash
python scripts/inference.py
```

预测结果输出到：

```text
result/output/
```

### 评估与可视化

```bash
python scripts/result.py
```

输出内容包括：

- `result/rmse.npy`
- `result/acc.npy`
- `result/loss.png`
- 指定日期和变量的预报对比图


# 官方来源与复现说明

- 模型实现来自官方 `NVIDIA/torch-harmonics`（BSD-3-Clause）中的 SFNO reference implementation。
- 当前案例目录抓取 commit：`49bac755cd8306fbd27a3604acafa65adf7ca202`（2026-08-14）。
- `conf/config.yaml` 默认使用小配置（`img_size=[32, 64]`、`embed_dim=16`、`num_layers=2`）用于连通性验证；论文级复现需按论文调整为 0.25°（721×1440）网格、26/73 通道与更大网络。
- 论文中以下细节论文未公开，复现时为假设项：天气模型内部频谱降采样倍数、位置嵌入形式、逐变量归一化统计区间、训练 batch size。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 SFNO 的独立工程整理与适配，模型源码参考自 Bonev 等人（2023）的官方 `torch-harmonics` 实现，遵循 BSD-3-Clause。
- 引用请参考：Bonev, Kurth, Hundt, Pathak, Baust, Kashinath, Anandkumar. Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere. ICML 2023.
