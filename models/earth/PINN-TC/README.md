<p align="center">
  <strong><span style="font-size: 30px;">PINN-TC</span></strong>
</p>

# 模型介绍

PINN-TC 用于从稀疏飞行观测、探空或模拟数据中重建热带气旋三维风场和气压相关场，通过物理方程约束未观测区域，主要服务于热带气旋涡旋初始化和数据同化研究。

论文：Realistic tropical cyclone wind and pressure fields can be reconstructed from sparse data using deep learning  
https://doi.org/10.1038/s43247-023-01144-2

# 模型描述

该方法由 Princeton University、California Institute of Technology、Stanford University 和 NOAA Geophysical Fluid Dynamics Laboratory 的研究团队提出。论文使用 Hurricane Ida 的 T-SHiELD 模拟场，以及 NOAA/AOML flight-level、dropsonde 和 Tail Doppler Radar 观测开展训练与验证。模型适用于稀疏热带气旋风场和位势场重建、物理一致性约束以及涡旋初始化研究。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 热带气旋场重建 | 从稀疏坐标观测重建风速、位势和压力垂直速度场。 |
| 物理信息约束 | 使用水平动量和压力连续方程约束未观测区域。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证稀疏采样、训练、完整网格推理、科学指标和可视化。 |
| 多卡训练 | 通过 `torchrun` 启动分布式数据并行训练。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/PINN-TC --local_dir ./PINN-TC
cd PINN-TC
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的流程验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活 DTK 及 Conda
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
pip install numpy pyyaml matplotlib
```

**GPU环境**

```bash
# 请首先激活 Conda
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
pip install numpy pyyaml matplotlib
```

### 训练数据介绍

训练数据为热带气旋模拟场和稀疏观测点，包含风暴中心坐标、水平风和位势信息。模型输入为 `[y,x,t,p]` 坐标，目标为同一位置的风场和位势，垂直速度通过物理方程约束。数据覆盖 `±400 km` 水平范围、`150-900 hPa` 压力层和 `-3/0/+3 h` 观测时刻。本仓库使用少量虚拟样本验证训练、推理和评估流程，不代表 T-SHiELD 或 NOAA 官方数据分布、训练规模或论文正式性能。

```bash
python scripts/fake_data.py --force
```

### 训练

单卡训练可使用：

```bash
python scripts/train.py
```

多卡训练可使用：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练同时优化观测数据损失和 beta-plane 水平动量方程、压力坐标连续方程的 PDE 残差。checkpoint 保存到 `result/checkpoints/pinn-tc.pt`，包含模型状态、模型配置、数据格式版本、Adam/L-BFGS 状态、训练历史和完成步数。

### 训练权重

本仓库不内置论文官方权重，公开资源中也没有可确认的官方 checkpoint。运行训练脚本可生成本地工程 checkpoint，但不声明与外部权重兼容，也不代表论文训练结果。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，根据稀疏热带气旋观测坐标和物理约束生成完整三维风场、位势和压力垂直速度场。完整数值结果保存到 `result/output/predictions.npz`。

### 评估和可视化

```bash
python scripts/result.py
```

评估计算风速 RMSE、Pearson 相关系数、PDE 残差和径向 RMSE，并保存三个观测时刻的整体聚合结果；当前评估不保存逐类别指标，也不单独保存逐时刻指标。评估同时生成稀疏观测、目标风速、预测风速和误差四联图；虚拟数据结果仅用于工程验证，不代表论文正式性能。

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 PINN-TC 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
