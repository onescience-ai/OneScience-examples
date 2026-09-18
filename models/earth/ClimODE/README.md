<p align="center">
  <strong>
    <span style="font-size: 30px;">ClimODE</span>
  </strong>
</p>

# 模型介绍

ClimODE 是芬兰阿尔托大学（Aalto University）等机构的研究者在2024年提出的一种气象预测模型。

论文：ClimODE: Climate and Weather Forecasting with Physics-informed Neural ODEs

https://arxiv.org/abs/2404.10024


# 模型描述

ClimODE 为物理信息神经常微分方程模型，用于全球、月尺度及区域气候与天气预报。模型将大气状态的时间演化表示为连续时间动力系统，并在神经 ODE 中结合输运形式的物理归纳偏置。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气预报 | 基于符合本项目五变量协议的 ERA5 数据训练或评估 ClimODE。 |
| 本地快速验证 | 使用虚拟 ERA5 HDF5 数据检查数据读取、训练、推理、评估和可视化流程。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动 PyTorch DistributedDataParallel。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 训练和推理必须使用 PyTorch 可识别的 GPU 或 DCU；CPU 可用于生成虚拟数据和检查配置，但不能运行当前训练与推理脚本。
- 多卡训练使用 NCCL 后端，请确保设备驱动、通信库和 PyTorch 版本匹配。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/ClimODE --local_dir ./ClimODE
cd ClimODE
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

OneScience 社区提供可供训练的 ERA5 数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认 `conf/config.yaml` 中数据路径设置正确：


```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

没有真实数据时，可生成保持真实 `721×1440` 原始网格的数据用于流程验证。虚拟数据不代表真实 ERA5，也不能用于复现论文指标：

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

默认最佳 checkpoint 保存为 `data/checkpoints/model_bak.pth`。


### 训练权重

本仓库在 weights/ 文件夹内提供基于 ERA5 再分析数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

推理默认优先读取 `data/checkpoints/model_bak.pth`：

```bash
python scripts/inference.py
```


### 评估和可视化

```bash
python scripts/result.py
```

脚本计算纬度加权 RMSE、ACC 和高斯预测分布 CRPS，并在 `result/output/figures/` 生成五个变量的预测、目标和误差图。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库是 ClimODE 论文的 OneScience 适配实现，并非 Aalto-QuML 官方发布版本。
