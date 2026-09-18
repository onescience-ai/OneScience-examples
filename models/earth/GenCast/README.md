<p align="center">
  <strong>
    <span style="font-size: 30px;">GenCast</span>
  </strong>
</p>

# 模型介绍

GenCast 是 Google DeepMind 提出的概率性全球天气预报模型，于2024年12月4日作为封面文章发表于顶级科学期刊《自然》。

论文：GenCast: Diffusion-based ensemble forecasting for medium-range weather

https://arxiv.org/abs/2312.15796

# 模型描述

Gencast 模型是基于图神经网络和扩散模型构建的集合预报成果，在多项评估中全面超越了欧洲中期天气预报中心（ECMWF）的顶级集合预报系统ENS。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气预报训练 | 使用符合 GenCast 数据协议的 ERA5 HDF5 数据训练模型。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、模型训练与推理、推理结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 JAX `pmap` 在同一主机的多张 GPU/加速卡上进行数据并行训练。 |



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
modelscope download --model OneScience/GenCast --local_dir ./GenCast
cd GenCast
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



### 训练

单卡：

```bash
# 若无真实数据，请先 python scripts/fake_data.py 创建虚拟数据。
python scripts/train.py 
```

多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --config conf/config.yaml --parallel-mode pmap --num-devices 2 --global-batch-size 2
# CUDA_VISIBLE_DEVICES 使用的显卡索引
# --num-devices 使用的显卡数量
# --global-batch-size batch_size大小（可被显卡数量整除）
```

训练后权重保存于 data/checkpoints/model_bak.npz


### 训练权重
本仓库在 weights/ 文件夹内提供官方预训练小权重。



### 推理

推理默认读取 `data/checkpoints/model_bak.npz`：

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

- 本仓库为 Gencast 原始论文的复现版本。
