<p align="center">
  <strong>
    <span style="font-size: 30px;">MeshGraphNet</span>
  </strong>
</p>

# 模型介绍

MeshGraphNet 是DeepMind提出的用于网格物理场建模的图神经网络，将非结构网格表示为图结构，通过网格边与空间邻近边上的消息传递学习节点状态的动态演化，并可在前向仿真过程中支持自适应网格离散。它适合流体绕流预测、结构力学变形仿真、布料/薄膜动力学建模、传统 CFD/FEM 高成本求解器加速等场景。


# 仓库说明

本仓库是 OneScience 整理的 MeshGraphNet 标准运行包，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

* 训练
* 推理
* 评估与可视化
* 生成空壳假数据用于流程连通性验证

当前不支持能力：
* 不内置预训练权重
* 不负责下载外部数据集进行适配


## 适用场景

| 场景 | 说明 |
|---|---|
| 流体绕流预测 | 基于网格节点预测速度、压力等流场变量 |
| 结构变形仿真 | 预测受力结构的位移、应力或形变过程 |
| 布料动力学模拟 | 模拟柔性薄膜、布料等可变形物体运动 |
| 非结构网格仿真 | 适用于复杂几何上的不规则网格数据 |
| 快速代理求解 | 替代部分 CFD/FEM 求解流程，提高推理效率 |
| 长时序物理预测 | 通过自回归方式逐步预测物理状态演化 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | OneCode 元信息 | 保持最小配置 |
| `config/config.yaml` | 训练、推理和数据配置 | 已适配本仓库相对路径 |
| `scripts/train.py` | 训练脚本 | 支持单卡和 torchrun 多卡 |
| `scripts/inference.py` | 推理脚本 | 需存在训练权重 |
| `scripts/result.py` | 评估与可视化脚本 | 读取 `result/output/*.npy` |
| `scripts/fake_data.py` | 假数据生成脚本 | 用于快速连通性验证 |
| `model/meshgraphnet.py` | 模型文件 | OneScience复现的经典TOP模型 |
| `weight/` | 权重目录 | 可放置预训练或发布权重 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。


## 3. 快速开始

### 安装运行环境

```bash
# 激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 生成假数据进行流程验证

如需先验证脚本、模型、checkpoint 和结果文件是否能够完整跑通，可使用仓库内置的 DGL 图假数据流程。默认配置已经开启假数据：

```yaml
datapipe:
  source:
    fake_data: true
    fake_data_path: data/cylinder_flow/fake_cylinder_flow.pt
```

执行以下命令会生成 `data/cylinder_flow/fake_cylinder_flow.pt`：

```bash
python scripts/fake_data.py
```
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
- 本仓库已保留相关来源及归属说明。使用、修改或分发本仓库内容时，请遵循相应的许可证要求。
