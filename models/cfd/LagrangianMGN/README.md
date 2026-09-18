<p align="center">
  <strong>
    <span style="font-size: 30px;">LagrangianMGN</span>
  </strong>
</p>

# 模型介绍
LagrangianMGN 是 DeepMind 相关团队提出的用于复杂物理系统仿真的图网络模型，主要用于流体、刚体和可变形材料等粒子系统动力学演化过程的快速预测。

论文：Learning to simulate complex physics with graph networks
https://arxiv.org/abs/2002.09405

# 模型描述
LagrangianMGN 基于图网络消息传递架构，使用粒子化物理仿真数据lagrangian进行训练，面向流体、刚体及可变形材料等复杂物理系统开展长时序动力学模拟。


## 适用场景

| 场景      |    说明                          |
| ------- | --------------------------- |
| 多材料交互仿真 | 处理流体、颗粒、刚体和可变形材料之间的相互作用     |
| 长时序物理预测 | 通过自回归 rollout 预测数百到数千步的系统演化 |
| 本地快速验证 | 使用虚拟数据检查数据读取、模型训练与推理、推理结果可视化 |


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
modelscope download --model OneScience/LagrangianMGN --local_dir ./LagrangianMGN
cd LagrangianMGN
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
OneScience社区提供可供训练的 DeepMind Lagrangian 数据，用户可通过下述命令下载，并确认'conf/config.yaml'中数据路径设置正确；

```bash
modelscope download --dataset OneScience/lagrangian --local_dir ./data/
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

### 训练权重
本仓库在weights/文件夹内提供基于DeepMind Lagrangian 数据训练的权重，该权重即将上传。

### 推理

```bash
python scripts/inference.py
```

推理脚本会从 `resume_dir` 读取 checkpoint，默认路径为 `weight/checkpoints`。

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
- LagrangianMGN 原始论文：[Learning to simulate complex physics with graph networks](https://arxiv.org/abs/2002.09405)。
- 本仓库已保留相关来源及归属说明。使用、修改或分发本仓库内容时，请遵循相应的许可证要求。
