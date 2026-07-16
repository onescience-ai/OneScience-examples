
---
<p align="center">
  <strong>
    <span style="font-size: 30px;">FactFormer</span>
  </strong>
</p>

# 模型介绍

FactFormer（Factorized Transformer）面向偏微分方程代理建模，将二维规则网格上的全局注意力分解为沿两个空间轴分别计算的注意力。相较于直接对全部网格点计算注意力，该结构显著降低了长序列建模的计算和显存开销，同时保留跨空间位置的信息交互能力。

本模型包用于二维 Kolmogorov Flow 涡量场时序预测。模型读取一段历史涡量场，通过 latent propagation 分块预测未来多个时间步，并以自回归方式完成完整时间窗口的推演。

论文：FactFormer: Factorized Transformer for Modeling Long-Range Dependencies in PDE Surrogate Modeling

# 仓库说明

本仓库是 OneScience 整理的 FactFormer 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 使用统一 YAML 配置训练和推理二维 FactFormer
- 使用 x、y 轴因子化注意力建模规则网格上的长程依赖
- 对 Kolmogorov Flow 涡量场执行多步分块自回归预测
- 使用内存映射读取大型 NumPy 数据，避免一次性载入全部数据
- 计算 MSE、相对 L2 误差并保存预测张量和可视化结果
- 在 CPU、GPU 或 DCU 上运行

当前不支持能力：

- 不内置约 9.4 GiB 的 Kolmogorov Flow 原始数据文件
- 随包权重仅用于流程验证，不代表完整训练后的模型精度
- 独立模型不包含原 OneScience 通用 `modules` 组件

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 涡量场时序预测 | 根据历史二维涡量场预测未来 Kolmogorov Flow 状态 |
| PDE 代理建模 | 学习规则网格上历史物理场到未来物理场的映射 |
| 因子化注意力研究 | 验证轴向注意力在高分辨率物理场建模中的效果 |
| 模型流程验证 | 通过小样本、单 epoch 配置检查训练和推理流程 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 数据、模型、训练及推理配置 | 路径相对于模型包根目录解析 |
| `model/factformer.py` | 独立 FactFormer 模型 | 不依赖 `onescience.modules` |
| `scripts/common.py` | 配置、设备、指标和分块 rollout 公共函数 | 训练和推理共用 |
| `scripts/train.py` | 训练与验证脚本 | 保存最佳相对 L2 检查点 |
| `scripts/inference.py` | 推理、评估与可视化脚本 | 读取 `weight/*.pt` |
| `data/` | Kolmogorov Flow 数据与统计量目录 | 原始 `.npy` 文件需单独准备 |
| `weight/` | 模型权重目录 | 含单 epoch 流程验证权重 |
| `result/` | 推理结果目录 | 首次推理时自动创建 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 完成正式训练。
- CPU 可用于缩小模型和数据后的流程验证，默认配置训练速度较慢。
- DCU 用户需要预先安装与当前集群匹配的 DTK 和 PyTorch 环境。

### 下载模型包

```bash
modelscope download --model OneScience/FactFormer --local_dir ./FactFormer
cd FactFormer
```

### 安装运行环境

已有 OneScience 环境时直接激活对应环境。新环境需安装 OneScience 及基础依赖：

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience torch numpy pyyaml matplotlib tqdm
```

## 3. 快速开始

### 准备数据

将二维 Kolmogorov Flow 数据文件放入 `data/`：

```text
data/
  kf_2d_re1000_256_120seed.npy
  km2d_stat.npz
```

原始数组应为 `float32`，形状如下：

```text
[120, 320, 256, 256]
```

其中四个维度依次表示轨迹、时间、x 网格和 y 网格。数据文件较大，数据管道默认通过 `numpy.load(..., mmap_mode="r")` 按需读取。

同时，OneScience社区提供可供训练的数据，用户可通过下述命令下载，并确认'conf/config.yaml'中数据路径设置正确；

```bash
modelscope download  --dataset OneScience/Kolmogorov_flow_2d --local_dir ./Kolmogorov_flow_2d
```

### 训练

```bash
python scripts/train.py
```

默认检查点保存至 `weight/factformer_kolmogorov.pt`。训练脚本完全由 `conf/config.yaml` 驱动，不接收命令行配置参数，也不读取环境变量覆盖配置。

### 推理、评估和可视化

```bash
python scripts/inference.py
```

脚本默认读取 `weight/factformer_kolmogorov.pt`，并在 `result/` 下生成：

```text
result/
  prediction_sample.pt
  prediction_sample.png
```

推理脚本同样完全由 `conf/config.yaml` 驱动，不接收命令行配置参数。权重位置由 `training.weight_dir` 和 `training.checkpoint_name` 共同确定，输出目录与文件名由 `inference` 配置控制。

# 配置说明

`conf/config.yaml` 分为五部分：

- `common`：设备和随机种子
- `datapipe`：数据文件、轨迹划分、时空降采样、归一化和 DataLoader
- `model`：隐藏维度、FactFormer 块、注意力头和 latent propagation 配置
- `training`：优化器、学习率调度、早停和权重目录
- `inference`：推理结果目录、文件名和保存样本数

模型的 `in_dim` 会根据 `t_in * out_dim` 自动更新，因此修改历史窗口时无需手工同步模型输入维度。

关键模型参数：

- `depth`：FactFormer Transformer 块数量
- `heads`：注意力头数，`hidden_dim` 必须能被其整除
- `mlp_ratio`：每个 Transformer 块中 MLP 隐层扩展倍率
- `latent_multiplier`：latent propagation 通道相对隐藏维度的倍率
- `max_latent_steps`：单次模型调用能够连续输出的最大时间步数
- `train_latent_steps`：训练时单次监督的未来时间步数

# 数据格式

Kolmogorov Flow `.npy` 文件标准形状为：

```text
[120, 320, 256, 256]
```

默认配置将空间下采样为 `128 x 128`，并在时间维每隔 2 帧采样。数据管道输出：

- `pos`：`[H*W, 2]`，周期区域 `[0, 2*pi)` 上的二维坐标
- `x`：`[H*W, t_in*out_dim]`，历史涡量场
- `y`：`[H*W, t_out*out_dim]`，未来涡量场

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Liu-Schiaffini et al. FactFormer: Factorized Transformer for Modeling Long-Range Dependencies in PDE Surrogate Modeling.
- Kolmogorov Flow 数据用于 Re=1000 二维涡量场预测任务。
- 本模型包采用 Apache-2.0 许可证，并保留原始论文和数据来源说明。
