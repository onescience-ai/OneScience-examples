<p align="center">
  <strong>
    <span style="font-size: 30px;">SEEDS</span>
  </strong>
</p>

# 模型介绍

SEEDS 是谷歌公司于2024年3月发布的一款生成式人工智能气象模型，全称“可扩展集成包络扩散采样器”，相关论文发表于权威期刊 Science Advances。

论文：SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models

https://arxiv.org/abs/2306.14066


# 模型描述

SEEDS 基于条件扩散模型，用于以少量数值天气预报种子成员为条件，高效生成大规模天气预报集合。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 集合天气预报研究 | 使用符合本项目立方球 NPZ 协议的数据训练条件扩散模型并生成预报集合。 |
| 本地快速验证 | 使用合成数据检查训练、推理、集合评估和可视化流程。 |
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
modelscope download --model OneScience/SEEDS --local_dir ./SEEDS
cd SEEDS
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

SEEDS 论文使用 GEFS reforecast 训练数据，并以 operational GEFS 成员作为条件、ERA5 作为评估参考。

没有真实数据时，可生成默认 `6×48×48` 网格的合成数据。合成数据只用于程序流程验证，不代表 GEFS、ERA5 或论文预报效果：

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

默认训练由 `conf/config.yaml` 中的 epoch 数控制，checkpoint 保存为 `data/checkpoint/model_bak.pth`。


### 训练权重

本仓库在 weights/ 文件夹内提供基于官方训练数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

推理默认读取 `data/checkpoint/model_bak.pth`，并按 `sampling.member_batch_size` 分块生成集合成员：

```bash
python scripts/inference.py
```


### 评估和可视化

```bash
python scripts/result.py
```

脚本计算集合均值 RMSE、ACC 和经验 CRPS，保存相应 NPY 指标，并生成 `result/forecast.png`；若训练损失文件存在，还会生成 `result/loss.png`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库是 SEEDS 论文的独立适配实现，不是 Google 官方产品。

