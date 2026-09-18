<p align="center">
  <strong>
    <span style="font-size: 30px;">EagleMeshTransformer</span>
  </strong>
</p>

# 模型介绍

EagleMeshTransformer 是法国里昂计算机科学研究实验室 LIRIS 提出的面向非结构动态网格流体预测的多尺度 Mesh Transformer 模型，主要适用于非定常湍流和长距离流场依赖的流体预测任务。

论文：[EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh Transformers](https://arxiv.org/abs/2302.10803)。

# 模型描述

EagleMeshTransformer 基于多尺度 Mesh Transformer 架构，使用eagle数据进行训练，面向复杂非定常流动开展速度场与压力场预测。
   


## 适用场景

| 场景 | 说明 |
|---|---|
| 非定常湍流预测 | 预测无人机、喷流、尾流等复杂非周期湍流中的速度场和压力场 |
| 非结构网格仿真 | 适用于复杂几何上的不规则网格数据 |
| CFD 代理求解加速 | 作为传统 Navier-Stokes / CFD 数值模拟的快速近似预测模型 |
| 长时序物理预测 | 通过自回归方式逐步预测物理状态演化 |

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
modelscope download --model OneScience/EagleMeshTransformer --local_dir ./EagleMeshTransformer
cd EagleMeshTransformer
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 假数据验证

默认配置已指向本仓库内的假数据目录，并将 `training.max_epoch` 设为 `1`，可直接生成最小 EAGLE NPZ 数据检查训练和推理流程。

```bash
python scripts/fake_data.py
```
### 训练数据介绍
OneScience 社区提供可供训练的 Eagle 数据，用户可通过下述命令下载，并确认'conf/config.yaml'中数据路径设置正确。

```bash
modelscope download --dataset OneScience/eagle --local_dir ./data
```

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练会在 `weight/` 下保存 `best_model.pth`。

### 训练权重
本仓库在weights/文件夹内提供EagleMeshTransformer预训练的权重，该权重即将上传。

### 推理

```bash
python scripts/inference.py
```

推理结果会保存至 `result/output/`。

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
- EagleMeshTransformer 原始论文：[EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh Transformers](https://arxiv.org/abs/2302.10803)。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。
