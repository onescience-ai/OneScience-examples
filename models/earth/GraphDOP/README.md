<p align="center">
  <strong>
    <span style="font-size: 30px;">GraphDOP</span>
  </strong>
</p>


# 模型介绍

GraphDOP（Graph-based Direct Observation Prediction）由 ECMWF 提出，是基于图神经网络（GNN）的端到端观测驱动（AI-DOP）天气预报模型。模型仅以地球系统观测（极轨/静止卫星亮温、掩星弯角、散射计后向散射、雷达高度计、探空/地面常规观测等）为输入与训练目标，不使用任何基于物理的（再）分析场，即可产生长达 5 天以上的中期预报。

论文：GraphDOP: Towards skilful data-driven medium-range weather forecasts learnt and initialised directly from observations

https://arxiv.org/abs/2412.15687

# 模型描述

GraphDOP 采用 编码器–处理器–解码器 结构：GNN 编码器把输入观测窗口内的观测按空间邻近关系映射到 O96（约 1°）潜网格上，Transformer 处理器在潜空间推进大气状态，GNN 解码器把潜网格映射回目标观测位置并逐通道输出预报。训练目标为按通道加权的最小均方误差（WMSE）。本仓库基于论文复现最小工程实现，并接入 OneScience 数据读取与训练流程。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 观测驱动的中期天气预报研究 | 从观测场直接学习大气状态表征并预报未来窗口。 |
| 图+Transformer 潜空间模型研究 | 编码器–处理器–解码器结构与 WMSE 目标的可复现实现。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、训练入口、推理和结果脚本。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |


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
modelscope download --model OneScience/GraphDOP --local_dir ./GraphDOP
cd GraphDOP
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

如需快速验证流程，可先运行虚拟数据脚本：

```bash
python scripts/fake_data.py
```

> 注：`scripts/fake_data.py` 根据模型输入/输出窗口与 `grid_shape` 生成 `[T, C, H, W]` 数据。由于 ERA5Datapipe 仅支持规则网格，本工程以 6 类观测网格通道近似论文中的不规则 Level-1 观测。

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练输出：

```text
data/checkpoints/model_bak.pth
data/checkpoints/trloss.npy
data/checkpoints/valoss.npy
```

### 训练权重
本仓库在 weight/ 文件夹内预留权重目录，默认不提供预训练权重，用户可依据论文配置自行训练。论文模型（潜通道 1024、O96 潜网格、64×H100 训练 70k 步）未公开发布权重。

### 推理

推理默认读取 `data/checkpoints/model_bak.pth`：

```bash
python scripts/inference.py
```

预测结果按预报帧时刻逐帧输出到：

```text
result/output/
```

### 评估与可视化

```bash
python scripts/result.py
```

输出内容包括：

- `result/rmse.npy`
- `result/acc.npy`
- `result/loss.png`
- 指定日期和变量的预报对比图


# 官方来源与复现说明

- 论文为 ECMWF AI-DOP 预印本，官方未开放实现（源码基于 PyTorch Geometric，动态观测图按批构建）。本仓库的 `model/graphdop.py` 为纯 PyTorch 最小复现，保持论文的 编码器–处理器–解码器 GNN 结构与 WMSE 目标。
- 与论文的差异（受 OneScience 网格化数据管线限制）：论文消费非规则的 Level-1 原始观测（每个观测点连接潜网格最近邻，动态构图）；本复现以 ERA5 网格化 h5 通道作为观测场占位，潜网格固定为规则 8 邻域图，边特征（方位角、Haversine 距离）与论文一致。观测随机丢弃（卫星 25%/常规 50%）等训练增强未实现。
- `conf/config.yaml` 默认使用小配置（32×32 网格、8×8 潜网格、latent_dim=64）用于连通性验证；论文级复现需调整为 O96 潜网格、latent_dim=1024 与更大的数据规模。
- 论文中以下细节论文未公开，复现时为假设项：逐通道权重 `w_{c,i}`（当前默认全 1）、图构建细节、处理器窗口化注意力的具体实现。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 GraphDOP 的独立工程复现（模型代码为原创最小实现），架构设计参考自 Alexe 等人（2024）的论文。
- 引用请参考：Alexe, M., E. Boucher, P. Lean, E. Pinnington, P. Laloyaux, A. McNally et al. GraphDOP: Towards skilful data-driven medium-range weather forecasts learnt and initialised directly from observations. arXiv:2412.15687, 2024.

