<p align="center">
  <strong>
    <span style="font-size: 30px;">WeatherNext2 · FGN</span>
  </strong>
</p>


# 模型介绍

FGN（**F**unctional **G**enerative **N**etworks，函数式生成网络）由 Google DeepMind 提出，是 WeatherNext 系列（WeatherNext2）中的概率全球天气预报模型。FGN 以单纯形边际分布（marginal）的公平连续排序概率分数（fair CRPS）为训练目标，通过在条件归一化层注入全局噪声向量的方式建模偶然（aleatoric）不确定性、以独立训练的模型集成建模认知（epistemic）不确定性，从而在仅优化逐位置边际目标的情况下仍能捕捉集合预报的联合空间结构，在 15 天中期预报上全面超越 GenCast 与 ECMWF ENS。

论文：Skilful joint probabilistic weather forecasting from marginals

https://arxiv.org/abs/2506.14285

# 模型描述

FGN 采用与 GenCast 去噪器一致的 GNN 编码器–处理器–解码器结构：稀疏 GNN 编码器把经纬网格输入映射到 6 次细分 icosahedral（球面二十面体）网格上的潜空间，graph-transformer 处理器在该网格的节点上推进大气状态，GNN 解码器把潜网格映射回输出网格。每次预报采一个 32 维全局噪声向量，经单次矩阵乘法嵌入后注入全部条件 LayerNorm 层（等价于对网络参数施加学习到的函数扰动），作为集合发散度的来源；模型按二阶马尔可夫假设（输入前两帧状态）以 6 小时步长自回归滚动生成预报。本仓库基于论文实现最小工程复现，并接入 OneScience 数据读取与训练流程。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 概率/集合中期天气预报研究 | 以 CRPS 为目标学习逐位置边际分布并生成联合集合预报。 |
| 不确定性建模方法研究 | 参数空间噪声注入（条件归一化）与深度集成机制的可复现实现。 |
| 图+Transformer 潜空间模型研究 | 编码器–处理器–解码器结构与公平 CRPS 目标。 |
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
modelscope download --model OneScience/WeatherNext2 --local_dir ./WeatherNext2
cd WeatherNext2
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

> 注：`scripts/fake_data.py` 根据二阶 Markov 输入、预报步数、batch size 与 `grid_shape` 生成 `[T, C, H, W]` 数据；当前小配置为 6 通道、32×32 网格。

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
本仓库在 weight/ 文件夹内预留权重目录，默认不提供预训练权重，用户可依据论文配置自行训练。论文模型（逐种子潜维度 768、24 层处理器、4 个模型种子集成、总计算量约 490 TPU·天）未公开发布权重。

### 推理

推理默认读取 `data/checkpoints/model_bak.pth`，对每个初始化时刻生成 `num_members` 个集合成员（各自独立采样全局噪声）并将成员均值作为确定性预报输出：

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

- 论文为 Google DeepMind 预印本（© 2025 Google DeepMind。All rights reserved.），官方未开放实现与权重。本仓库的 `model/fgn.py` 为纯 PyTorch 最小复现，保持论文的 编码器–处理器–解码器 GNN 结构、条件 LayerNorm 全局噪声注入与公平 CRPS 目标（训练时 N=2 集合样本）。
- 与论文的差异（受 OneScience 网格化数据管线与连通性验证规模限制）：论文使用 6 次细分 icosahedral 潜网格（约 40k 节点）与 0.25°（1440×721）输出网格，单种子约 180M 参数；本复现的潜网格为固定规则 8 邻域网格（`mesh_shape`），输入为 ERA5 网格化 h5 通道占位，潜维度默认 64。潜网格节点数与边特征由 `mesh_shape` 与 `channel_weights` 配置决定。
- 论文中以下细节论文未公开，复现时为假设项：条件 LayerNorm 中噪声到各层 scale/shift 的具体投影形式（当前用逐层线性投影，初始化置零以保证初始为标准 LayerNorm）；多任务损失的逐通道权重（当前默认全 1）；概率输出以集合成员均值作为确定性预报的评估口径。
- 论文级复现需按论文四阶段训练（ERA5 1°12h → 1°6h → 0.25°6h → HRES-fc0 0.25° AR 微调），`conf/config.yaml` 默认使用小配置用于连通性验证。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 FGN（WeatherNext2）的独立工程复现（模型代码为原创最小实现），架构设计参考自 Alet 等人（2025）的论文。
- 引用请参考：Alet, F., Price, I., El-Kadi, A., Masters, D., Markou, S., Andersson, T. R., Stott, J., Lam, R., Willson, M., Sanchez-Gonzalez, A. and Battaglia, P. Skilful joint probabilistic weather forecasting from marginals. arXiv:2506.14285, 2025.

