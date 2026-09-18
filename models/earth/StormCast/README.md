<p align="center">
  <strong>
    <span style="font-size: 30px;">StormCast</span>
  </strong>
</p>

# 模型介绍

StormCast 是 NVIDIA 提出的生成式区域天气预报模型，面向中尺度对流天气的高分辨率短临预测。

论文：StormCast: A Machine Learning Method for Meso-β-Scale Convection-resolving Weather Forecasting

https://arxiv.org/abs/2408.10958

# 模型描述

StormCast 模型使用大尺度天气背景约束区域状态演变，并通过生成式扩散方法补充确定性预报难以刻画的细尺度结构。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 两阶段天气预报训练 | 依次训练确定性回归模型和条件残差扩散模型。 |
| 本地快速验证 | 使用虚拟数据检查数据读取，模型训练、推理、推理结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

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
modelscope download --model OneScience/StormCast --local_dir ./StormCast
cd StormCast
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


### 生成虚拟数据进行流程验证


```bash
python scripts/fake_data.py
```

虚拟数据只用于检查数据协议和程序流程，不代表模型的科学预报能力。

### 训练

单卡：

```bash
# 训练确定性回归模型，默认权重保存至 data/checkpoint/regression/model_bak.pt
python scripts/train.py --stage regression
# 训练残差扩散模型，默认权重保存至 data/checkpoint/diffusion/model_bak.pt
python scripts/train.py --stage diffusion
```


### 多卡

```bash
# 训练确定性回归模型
torchrun --nproc_per_node=2 scripts/train.py --stage regression
# 训练残差扩散模型
torchrun --nproc_per_node=2 scripts/train.py --stage diffusion
```


### 训练权重

本仓库在 weights/ 文件夹内提供基于 ERA5 再分析数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

使用默认配置和两个训练后保存的权重运行自回归预测：

```bash
python scripts/inference.py
```

### 评估和可视化


```bash
python scripts/result.py
```

图片默认保存至 `outputs/inference/plots/`


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 StormCast 原始论文的复现版本。
