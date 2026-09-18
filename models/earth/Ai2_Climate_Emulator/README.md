<p align="center">
  <strong>
    <span style="font-size: 30px;">Ai2_Climate_Emulator</span>
  </strong>
</p>

# 模型介绍

Ai2 Climate Emulator（ACE）是 Allen Institute for AI（AI2）提出的全球大气状态模拟器。

论文：ACE: A fast, scalable foundation model for the atmosphere

https://arxiv.org/abs/2310.02074

# 模型描述

本项目使用 PyTorch 和 `torch_harmonics` 实现球面 Fourier Neural Operator（SFNO）前向图，以当前 6 小时大气状态和外部强迫为输入，预测下一时刻状态，并可通过自回归方式生成多步气候/天气场。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球大气状态模拟 | 使用符合 ACE 40/44 通道协议的 FV3GFS 数据训练一步预测模型。 |
| 本地快速验证 | 使用 `scripts/fake_data.py` 生成合成 NPZ，检查训练、推理和结果可视化流程。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动 PyTorch DDP。 |

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
modelscope download --model OneScience/Ai2_Climate_Emulator --local_dir ./Ai2_Climate_Emulator
cd Ai2_Climate_Emulator
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

ACE 原论文使用 FV3GFS 产生的 11 成员初始条件集合模拟：10 个成员用于训练、1 个成员用于验证，并以 6 小时频率输出后重网格到 Gaussian 经纬网格。原始 FV3GFS 文件和 NOAA `fregrid` 不包含在本模型包中，需要用户自行准备并转换为项目要求的 NPZ：

```text
inputs:  [N, 40, H, W]
targets: [N, 44, H, W]
```

没有真实数据时，可生成仅用于流程验证的合成数据：

```bash
python scripts/fake_data.py
```

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 scripts/train.py
```

训练检查点默认写入 `data/checkpoint/model_bak.pt`。

### 训练权重
本仓库在 weights/ 文件夹内提供基于 FV3GFS 数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理结果默认保存为 `output/infer/rollout.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

默认输出面积加权 RMSE、全球均值偏差和 PNG 图像到 `output/pic/`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 ACE 模型的复现版本。
