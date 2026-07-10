<p align="center">
  <strong>
    <span style="font-size: 30px;">EagleMeshTransformer</span>
  </strong>
</p>

# 模型介绍

EagleMeshTransformer 是法国里昂计算机科学研究实验室 LIRIS 提出的面向非结构动态网格流体预测的多尺度 Mesh Transformer 模型。该模型通过几何聚类和图池化将原始网格压缩为更粗尺度的 token，并在粗尺度表示上引入全局多头自注意力机制，以较低的计算复杂度捕获长距离依赖关系；随后，模型通过解码器将特征上采样回原始网格分辨率，用于预测下一时刻或未来时刻的速度场和压力场。该模型主要适用于非定常湍流、动态重网格、复杂边界几何以及存在长距离流场依赖的流体预测任务。



# 仓库说明

本仓库是 OneScience 整理的 EagleMeshTransformer 标准运行包，面向 OneCode 自动化运行和本地快速验证场景。

当前支持能力：

* 训练
* 推理
* 推理结果摘要
* 生成最小 NPZ 假数据用于流程连通性验证

当前不支持能力：
* 不内置预训练权重
* 不负责下载外部数据库进行适配


## 适用场景

| 场景 | 说明 |
|---|---|
| 非定常湍流预测 | 预测无人机、喷流、尾流等复杂非周期湍流中的速度场和压力场 |
| 动态网格流体仿真 | 适用于源体运动、边界变化或需要重网格的流体场景 |
| 长距离气流依赖建模 | 通过粗尺度全局注意力捕获远距离涡流、尾迹和流场传播关系 |
| 非结构网格仿真 | 适用于复杂几何上的不规则网格数据 |
| CFD 代理求解加速 | 作为传统 Navier-Stokes / CFD 数值模拟的快速近似预测模型 |
| 长时序物理预测 | 通过自回归方式逐步预测物理状态演化 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `configuration.json` | OneCode 元信息 | 保持最小配置 |
| `config/config.yaml` | 训练、推理和数据配置 | 已适配本仓库相对路径 |
| `scripts/train.py` | 训练脚本 | 支持单卡和 torchrun 多卡 |
| `scripts/inference.py` | 推理脚本 | 需存在训练权重 |
| `scripts/result.py` | 推理结果摘要脚本 | 读取 `result/output/prediction_*.npy` |
| `scripts/fake_data.py` | 假数据生成脚本 | OneScience复现的经典TOP模型 |
| `model/graphViT.py` | 模型文件 | 依赖 OneScience 源码包 |
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
git clone https://gitee.com/onescience-ai/onescience onescience
cd onescience 
bash install.sh cfd
```

### 假数据验证

默认配置已指向本仓库内的假数据目录，并将 `training.max_epoch` 设为 `1`，可直接生成最小 EAGLE NPZ 数据检查训练和推理流程。

```bash
python scripts/fake_data.py
```

如需使用真实 Eagle 数据，请准备数据集并根据本地实际目录修改 `config/config.yaml`。


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

### 推理

```bash
python scripts/inference.py
```

推理结果会保存至 `result/output/`。

### 评估和可视化

```bash
python scripts/result.py
```

`result.py` 会读取 `result/output/prediction_*.npy`，打印预测文件数量、数组形状、均值和标准差。当前脚本不生成 GIF 可视化，也不计算真实 EAGLE 评估指标。


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证
- EagleMeshTransformer 原始论文：[EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh Transformers](https://arxiv.org/abs/2302.10803)。
- 本仓库保留来源说明，并面向 OneScience 自动运行场景进行整理。
