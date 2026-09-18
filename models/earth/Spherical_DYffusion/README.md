<p align="center">
  <strong>
    <span style="font-size: 30px;">Spherical DYffusion</span>
  </strong>
</p>

# 模型介绍

Spherical DYffusion 由 Salva Rühling Cachay 等人提出，用于全球气候模式的概率模拟。

论文：Probabilistic Emulation of a Global Climate Model with Spherical DYffusion

https://arxiv.org/abs/2406.14798


# 模型描述

Spherical DYffusion 使用 SFNO 建模球面动力学，并采用 DYffusion 的插值器与 forecaster 两阶段训练方法实现概率集合模拟。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 本地流程验证 | 使用虚拟 37 通道全球网格数据检查训练、推理和可视化链路。 |
| FV3GFS 协议检查 | 校验 NetCDF 变量、空间维度和连续时间帧。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载并运行当前精简实现。 |
| 多卡训练 | 通过 `torchrun` 启动 PyTorch DistributedDataParallel。 |


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
modelscope download --model OneScience/Spherical_DYffusion --local_dir ./Spherical_DYffusion
cd Spherical_DYffusion
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

### 生成测试数据

```bash
python scripts/fake_data.py
```

默认生成 `data/data/synthetic_fv3gfs.nc` 和 `synthetic_fv3gfs.json`。数据包含 37 个协议变量：地面气压、地表温度、8 层气温、8 层总水、8 层东西风、8 层南北风，以及 `DSWRFtoa`、`HGTsfc` 和 `ocean_fraction`。这些确定性合成场只用于协议检查。


### 训练

单卡：

```bash
python scripts/train.py
```


多卡：

```bash
torchrun --nproc_per_node=8 scripts/train.py
```

训练默认从随机初始化开始，保存 `data/checkpoint/model_bak.pt` 和 `last.pt`。



### 官方权重

本仓库在 weights/ 文件夹内提供基于 FV3GFS 数据训练的权重，权重文件即将上传，预计将于近期完成。


### 推理

```bash
python scripts/inference.py
```


### 评估和可视化

```bash
python scripts/result.py
```


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 Spherical DYffusion 原始论文的复现版本。

