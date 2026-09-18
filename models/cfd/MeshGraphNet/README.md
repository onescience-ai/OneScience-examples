<p align="center">
  <strong>
    <span style="font-size: 30px;">MeshGraphNet</span>
  </strong>
</p>

# 模型介绍

MeshGraphNets 是 DeepMind 团队提出的基于计算网格的图神经网络物理仿真模型，可对流体、结构和布料等复杂物理系统的动力学演化进行快速预测。

论文：Learning Mesh-Based Simulation with Graph Networks
https://arxiv.org/abs/2010.03409

# 模型描述
MeshGraphNets 基于编码器—处理器—解码器式图网络架构，使用流体力学、结构力学和布料仿真轨迹数据进行训练，面向复杂物理系统开展长时序动力学模拟。

## 适用场景

| 场景 | 说明 |
|---|---|
| 流体绕流预测 | 基于网格节点预测速度、压力等流场变量 |
| 结构变形仿真 | 预测受力结构的位移、应力或形变过程 |
| 布料动力学模拟 | 模拟柔性薄膜、布料等可变形物体运动 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本|


# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。


### 下载模型包

```bash
modelscope download --model OneScience/MeshGraphNet --local_dir ./MeshGraphNet 
cd MeshGraphNet 
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

### 训练数据介绍

OneScience 社区提供可供训练的 `cylinder_flow` 数据，用户可通过下述命令下载，并确认 `config/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/cylinder_flow --local_dir ./data
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

训练会在 `weight/checkpoints` 下保存 pth文件。

### 训练权重
本仓库在weights/文件夹内提供基于 cylinder_flow 数据集训练的权重，即将上传该权重。

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
- MeshGraphNet 原始论文：[Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。

