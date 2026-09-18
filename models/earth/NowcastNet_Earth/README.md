<p align="center">
  <strong>
    <span style="font-size: 30px;">NowcastNet_Earth</span>
  </strong>
</p>

# 模型介绍

NowcastNet 是一个由清华大学团队提出的极端降水临近预报大模型，其研究成果发表于《自然》(Nature)正刊。

论文：Skilful nowcasting of extreme precipitation with NowcastNet

https://www.nature.com/articles/s41586-023-06184-4

# 模型描述

NowcastNet 将数据驱动的深度学习与基于物理方程的数值方法融合在一个统一的框架中，通过两个核心网络协同工作，分别处理不同尺度的降水过程。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 短临降水预报训练 | 使用 MRMS 数据训练 NowcastNet 模型。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、模型训练与推理、推理结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过  `torchrun` 在同一主机的多张 GPU/加速卡上进行数据并行训练。 |



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
modelscope download --model OneScience/NowcastNet --local_dir ./NowcastNet
cd NowcastNet
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

### 生成虚拟数据

虚拟数据只用于检查数据协议和程序流程，不代表真实 MRMS 数据或模型预报效果。

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

训练后权重默认保存于 data/checkpoints/ 目录。


### 训练权重
本仓库在 weights/ 文件夹内提供基于 MRMS 数据训练的权重，权重文件即将上传，预计将于近期完成。



### 推理

推理默认读取 `data/checkpoints/` 目录下的训练权重：

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

- 本仓库为 NowcastNet 原始论文的复现版本。
