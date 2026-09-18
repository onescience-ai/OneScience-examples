<p align="center">
  <strong>
    <span style="font-size: 30px;">GLONET</span>
  </strong>
</p>

# 模型介绍

GLONET（Global Ocean Neural Network）是由欧洲领先的海洋预报中心Mercator Ocean International开发的全球海洋神经网络预报系统。

论文：GLONET: Mercator's end-to-end neural Global Ocean forecasting system

https://arxiv.org/abs/2412.05454


# 模型描述

GLONET 是面向全球海洋状态预报的神经网络模型，模型接收连续两个日状态，输出下一日的 34 通道海洋状态。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球海洋预报研究 | 使用 GLORYS12 兼容数据训练 FNO/CNN 双分支海洋预报模型。 |
| 本地快速验证 | 使用合成海洋场检查数据读取、预训练、微调、推理和可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 进行多卡训练。 |

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
modelscope download --model OneScience/GLONET --local_dir ./GLONET
cd GLONET
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

官方工作使用 GLORYS12 再分析数据。真实数据必须先转换为 `conf/config.yaml` 中的通道顺序和网格；项目不随包提供 GLORYS12 原始数据。默认合成数据仅用于接口检查：

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

检查点默认保存到 `data/checkpoints/` 目录。

### 训练权重

本仓库在 weights/ 文件夹内提供基于 GLORYS12 数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

预测张量默认写入 `result/glonet/data/prediction.pt`。

### 评估和可视化

```bash
python scripts/result.py
```

默认生成 `result/glonet/prediction.png`；只有提供真实参考场时才会计算有意义的误差。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 GLONET 原始论文的复现版本。
