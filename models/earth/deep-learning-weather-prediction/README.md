<p align="center">
  <strong><span style="font-size: 30px;">DLWP-CS</span></strong>
</p>

# 模型介绍

DLWP-CS 使用 cubed-sphere 卷积网络进行全球天气预报，降低普通经纬网在极区带来的几何问题。

论文：Improving Data-Driven Global Weather Prediction Using Deep Convolutional Neural Networks on a Cubed Sphere

https://doi.org/10.1029/2020MS002109

# 模型描述

当前目录根据论文和官方代码实现独立 PyTorch 结构 smoke 版本，包含 cubed-sphere 跨面 padding、卷积、简化 U-Net、capped leaky ReLU 和自回归推理。它不是论文实验架构或 ERA5 训练复现。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| Cubed-sphere 结构研究 | 验证六面邻接、翻转和卷积。 |
| 本地快速验证 | 使用 fake 数据完成训练和 rollout。 |
| ERA5 全球天气预报 | 后续可接入 Tempest-Remap 处理后的 ERA5。 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `download.sh` | ModelScope 资源下载脚本 | 下载官方源码使用的 GRIB 参数映射表 |
| `conf/config.yaml` | 训练配置 | 定义样本规模、优化器参数、设备和 checkpoint 路径 |
| `model/model.py` | Compact DLWP-CS 模型 | 实现 cubed-sphere U-Net、激活函数、loss 和 rollout |
| `model/topology.py` | Cubed-sphere 拓扑算子 | 实现跨面 padding 和分面卷积 |
| `model/dataset.py` | 训练数据集封装 | 组织合成 cubed-sphere 输入与目标 |
| `model/fake_data.py` | 合成数据生成模块 | 为结构验证生成六面球测试数据 |
| `scripts/train.py` | 训练入口 | 执行多 epoch 训练、验证和断点续训 |
| `scripts/inference.py` | 自回归推理入口 | 加载 checkpoint 并执行多步 rollout |
| `scripts/result.py` | 评估与可视化入口 | 输出 RMSE、ACC 和分面结果图 |
| `official-source/` | DLWP-CS 官方代码快照 | 包含官方数据处理、重映射、模型和教程 |
| `weight/` | 本地训练权重目录 | 保存训练产生的 checkpoint |

# 使用说明

## 1. OneCode 使用

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- CPU 可运行当前最小配置。
- 真实数据训练推荐使用 GPU。

## 3. 快速开始
### 安装运行环境

**DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 数据

当前默认训练使用 `model/dataset.py` 的确定性 fake Dataset，不需要额外下载；每个样本为 `[C,6,H,W]`，验证集使用独立 seed。

### 训练

```bash
python scripts/train.py
```

脚本执行多 epoch 训练、验证、学习率调度和 early stopping：

1. 按索引生成 `[C,6,H,W]` fake 数据；
2. 检查六面 topology 和 capped leaky ReLU；
3. 执行每个 epoch 的 U-Net forward、MSE、backward 和优化；
4. 在独立 fake validation Dataset 上计算验证损失；
5. 保存 latest/best checkpoint 与 history，并支持 `--resume`。

```bash
python scripts/train.py --epochs 10
python scripts/train.py --resume weight/training/latest.pth --epochs 20
```

checkpoint 输出：

```text
weight/model.pth
weight/training/latest.pth
weight/training/best.pth
weight/training/history.json
```

### 推理

```bash
python scripts/inference.py
```

推理结果：

```text
result/prediction.pt
result/target.pt
result/inference.json
```

### 结果检验

```bash
python scripts/result.py
```

结果脚本生成 `result/metrics.json` 和 `result/comparison.png`。当前 `rmse` 和 `spatial_acc` 只用于 fake tensor 连通性检查，未执行论文要求的反归一化、纬度面积加权、cubed-sphere 逆映射或逐日气候态 anomaly。

### 论文与当前实现的 I/O

| 项目 | 论文 DLWP-CS | 当前 smoke 实现 |
| --- | --- | --- |
| 动态输入 | 4 变量的 `t-6h,t`，8 通道 | 无物理语义的 2 通道单状态 |
| 辅助输入 | 太阳辐射、海陆掩膜、地形 | 未实现 |
| 空间网格 | `[6,48,48]` cubed sphere | `[6,8,8]` fake grid |
| 输出 | 4 变量的 `t+6h,t+12h`，8 通道 | 同形 2 通道输出 |
| 网络/训练 | 两级 U-Net，两次自回归联合损失 | 一级简化 U-Net，多 epoch 单步 MSE 训练 |
| 分析 | 物理单位、纬度加权 RMSE/ACC | 无物理单位 smoke 指标 |

完整执行顺序为 `train.py -> inference.py -> result.py`。fake Dataset 保持 cubed-sphere 输入 shape，但不代表连续天气时间序列；模型包发布时不携带本地训练权重或 `result/` 生成物。真实模式还需实现 ERA5 Dataset、CS48 remap、4 个动态变量、辅助场、归一化统计量和论文的两次迭代训练损失。

### 真实数据

真实训练需要 ERA5 的 Z500、Z1000、300-700 hPa 位势厚度和 2 米温度，以及太阳辐射、海陆掩膜、地形和 Tempest-Remap 离线映射权重。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 官方代码：https://github.com/jweyn/DLWP-CS
- 本目录依据论文和官方结构进行独立适配，遵循 GPL-3.0。
