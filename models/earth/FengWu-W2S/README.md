<p align="center">
  <strong>
    <span style="font-size: 30px;">FengWu-W2S</span>
  </strong>
</p>

# 模型介绍

FengWu-W2S（FengWu Weather-to-Subseasonal）是基于 FengWu 扩展的无缝天气到次季节全球预报模型。

论文：FengWu-W2S: A deep learning model for seamless weather-to-subseasonal forecast of global atmosphere

https://arxiv.org/abs/2411.10191

# 模型描述

该模型采用 6 小时步长进行最长 42 天自回归预报，并通过大气、海洋和陆面耦合以及多样性扰动增强延伸期预报能力。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气到次季节预报研究 | 使用 78 通道、6 小时时间分辨率的全球气象和海气陆耦合数据训练模型。 |
| 本地快速验证 | 使用小网格合成 HDF5 数据检查训练、微调、推理和推理输出可视化流程。 |
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
modelscope download --model OneScience/FengWu-W2S --local_dir ./FengWu-W2S
cd FengWu-W2S
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

真实训练数据需要包含 conf/config.yaml 中列出的全部通道。用于快速检查时，可显式生成小网格合成数据，同于流程测试：

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
torchrun --nproc_per_node=2 scripts/train.py
```

基础训练检查点默认保存到 data/checkpoints/model_bak.pth。


### 训练权重

本仓库在 weights/ 文件夹内提供基于ERA5再分析数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理需要项目训练产生的检查点，默认将逐时效预测保存到 result/output/<year>/，并写入 result/output/index.json。

### 评估和可视化

```bash
python scripts/result.py
```

脚本计算逐通道 RMSE、归一化 RMSE 和去气候态 ACC，并生成预报对比、时效技能、通道排序和训练损失图。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- FengWu-W2S 论文：https://arxiv.org/abs/2411.10191 。
- 本目录是论文方法的独立复现，不代表论文作者发布的官方代码、权重或训练结果。
