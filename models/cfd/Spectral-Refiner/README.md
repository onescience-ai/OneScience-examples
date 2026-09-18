<p align="center">
  <strong>
    <span style="font-size: 30px;">Spectral-Refiner</span>
  </strong>
</p>

# 模型介绍

Spectral-Refiner 是面向时空 Fourier Neural Operator 的物理残差微调方法。本项目对应论文 Table 2 的二维强迫湍流实验：先在 64x64 涡量轨迹上训练 SFNO，再在 256x256 分辨率上使用 Navier-Stokes PDE residual 的 H^-1 负 Sobolev 范数微调谱输出层。

论文：[Spectral-Refiner: Accurate Fine-Tuning of Spatiotemporal Fourier Neural Operator for Turbulent Flows](https://arxiv.org/abs/2405.17211)

# 模型描述

模型输入前 10 个时间步的二维涡量场，预测后续 40 个时间步。基础 SFNO 使用 4 层时空谱算子，空间模态数为 12x12、时间模态数为 5、隐藏宽度为 20。Spectral-Refiner 冻结基础网络，将输出谱层扩展到 64x64x6，并以 H^-1 PDE residual 微调 50 步。本项目由 OneSkill技能 独立进行论文复现整理而成。

## 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 二维湍流预测 | 根据历史涡量场预测后续时空演化 |
| PDE 代理模型 | 使用 Fourier Neural Operator 加速周期边界流体问题求解 |
| 物理残差微调 | 使用负 Sobolev 范数约束预测结果的 PDE residual |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 完成正式训练和推理。
- CPU 可用于导入和 `--smoke` 连通性检查。
- 本次正式验证使用 PyTorch 2.5.1 和 1 张 DCU。

### 进入模型目录

```bash
cd Spectral-Refiner
```

### 安装运行环境

**DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

OneScience 社区在 ModelScope 的 [`OneScience/fno`](https://modelscope.cn/datasets/OneScience/fno) 数据集中提供 Spectral-Refiner 实验所需的 Navier–Stokes 涡量轨迹，可通过以下命令下载：

```bash
modelscope download \
  --dataset OneScience/fno \
  --local_dir ./data
```
相关数据位于下载目录的 data/ 子目录：
```
data/
├── fnodata_extra_64x64_N1280_v1e-3_T50_steps100_alpha2.5_tau7.pt
└── fnodata_extra_fp64_256x256_N16_v1e-3_T50_steps100_alpha2.5_tau7.pt
```
请确保 config/config.yaml 中的 data.train_file 和 data.test_file 分别指向上述两个文件，实验默认从低分辨率数据中选取 1152 条轨迹训练、128 条轨迹验证，每次使用连续 10 个时间步预测后续 40 个时间步；高分辨率数据用于 256×256 网格上的模型评测和 $H^{-1}$ 物理残差谱微调。


### 训练
默认配置 `config/config.yaml` 对应论文主实验设置。

```bash
python scripts/train.py --config config/config.yaml
```

### 训练权重

`weight/best_model.pt` 同时保存基础 SFNO、Spectral-Refiner 输出层、配置和权重选择指标，可由推理脚本严格加载。


### 推理

```bash
python scripts/inference.py
```

推理脚本严格加载 `weight/best_model.pt`，结果保存到：

```text
./results/predictions.pt
```

### 评估

```bash
python scripts/result.py
```

评估结果会打印到终端并保存到：

```text
./results/metrics.json
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Spectral-Refiner 论文：[arXiv:2405.17211](https://arxiv.org/abs/2405.17211)。
- 论文公开实现：[scaomath/torch-cfd](https://github.com/scaomath/torch-cfd)。
- 本项目按 MIT License 的来源要求保留论文与公开实现信息。

