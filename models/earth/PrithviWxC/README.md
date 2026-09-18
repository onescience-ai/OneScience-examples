<p align="center">
  <strong>
    <span style="font-size: 30px;">Prithvi WxC</span>
  </strong>
</p>


# 模型介绍

Prithvi WxC（Weather and Climate）由 NASA-IMPACT 与 IBM 等团队联合提出，是基于视觉 Transformer（Hiera + MaxViT 交替的本地/全局注意力）的天气与气候基础模型，支持预报（6 小时步长滚动）与气候模拟（内部误差增长）两类任务。

论文：Prithvi WxC: Foundation Model for Weather and Climate

https://arxiv.org/abs/2409.13598

# 模型描述

Prithvi WxC 是确定性的全球天气基础模型：输入连续两个 6 小时时刻的大气状态（叠加可选的静态场），输出目标时刻的状态，长时效通过自回归滚动获得。本仓库基于官方 `NASA-IMPACT/Prithvi-WxC` 实现整理，并接入 OneScience 数据读取与训练流程。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气与气候基础模型研究 | 基于 ERA5 数据训练/微调视觉 Transformer 预报模型。 |
| 长时间自回归滚动 | 以 6 小时为步长自回归生成中长期预报。 |
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
- 论文级配置（embed_dim=2560、25 编码块、5 解码块，约 23 亿参数）需要大规模显存。


### 下载模型包

```bash
modelscope download --model OneScience/PrithviWxC --local_dir ./PrithviWxC
cd PrithviWxC
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

> 注：`scripts/fake_data.py` 根据模型配置生成两个输入时刻所需的 `[T, C, H, W]` HDF5 数据，并生成 `data/static/static.npy`（当前为 `[4, 32, 64]`）供训练与推理加载。

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
本仓库在 weight/ 文件夹内预留权重目录。官方在 Hugging Face 上发布约 23 亿参数权重（如 PrithviWxC_160_13b_2t_0p5d_v1.pt），与本仓库小配置结构不一致，加载前需自行对齐通道数/网格大小；本仓库默认不提供权重，用户可依据论文配置自行训练。

### 推理

推理默认读取 `data/checkpoints/model_bak.pth`：

```bash
python scripts/inference.py
```

预测结果输出到：

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

- 模型实现来自官方 `NASA-IMPACT/Prithvi-WxC`（MIT License），官方实现以 `model/prithvi_wxc_official.py` 原样内嵌，`model/prithvi_wxc.py` 仅为 YAML 驱动的薄包装（恒等归一化参数，便于小配置连通性验证）。
- 当前案例目录抓取 commit：`79dabfcd17abe77e2d5c696707c0164a04f2ec01`（2026-02-05）。
- `conf/config.yaml` 默认使用小配置（`embed_dim=32`、`n_blocks_encoder=1`、`n_blocks_decoder=1`）用于连通性验证；论文级复现需按论文调整为 0.5°×0.625° 网格、160 通道、`embed_dim=2560` 与 13+12 编码块/3+2 解码块。
- 论文中以下细节论文未公开，复现时为假设项：数据归一化统计量（当前使用恒等归一化，真实统计量随数据接入）、掩码训练细节与预训练调度、部分超参数（如相对位置编码实现）。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 Prithvi WxC 的独立工程整理与适配，模型源码参考自 Schmude 等人（2024）的官方 `NASA-IMPACT/Prithvi-WxC` 实现，遵循 MIT License。
- 引用请参考：Schmude et al. Prithvi WxC: Foundation Model for Weather and Climate. arXiv:2409.13598, 2024.

