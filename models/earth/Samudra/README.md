<p align="center">
  <strong>
    <span style="font-size: 30px;">Samudra</span>
  </strong>
</p>

# 模型介绍

Samudra 是由 M2LInES 团队开发的全球海洋模拟器。

论文：Samudra: An AI Global Ocean Emulator for Climate

https://doi.org/10.1029/2024GL114318


# 模型描述

Samudra 在约 1 度全球海洋网格上以 5 天为时间步长进行预测，主要用于通过深度学习模型模拟海洋环流模式 OM4 的演化。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球海洋模拟 | 使用符合 Samudra 77 状态通道和 4 强迫通道协议的 OM4 数据训练模型。 |
| 本地快速验证 | 使用合成 NPZ 数据检查训练、推理和海洋场可视化流程。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装依赖并运行脚本。 |
| 多卡训练 | 通过 torchrun 启动 PyTorch DistributedDataParallel。 |

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
modelscope download --model OneScience/Samudra --local_dir ./Samudra
cd Samudra
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

官方训练数据来自 NOAA/GFDL OM4 模拟，M2LInES 提供数据发布：

https://huggingface.co/datasets/M2LInES/Samudra-OM4

没有真实 OM4 数据时，可生成仅用于流程验证的合成数据：

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

项目检查点默认保存为 data/checkpoints/model_bak.pth。

### 训练权重


本仓库在 weights/ 文件夹内提供基于 OM4 数据训练的权重，权重文件即将上传，预计将于近期完成

### 推理

```bash
python scripts/inference.py
```

预测默认保存为 result/output/prediction.npz。

### 评估和可视化

```bash
python scripts/result.py
```

默认生成 result/forecast_maps.png 和 result/temperature_profile.png。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Samudra 论文：https://doi.org/10.1029/2024GL114318 。
- 本仓库为 Samudra 原始论文的复现版本。
