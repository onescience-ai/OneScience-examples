<p align="center">
  <strong>
    <span style="font-size: 30px;">NowCastNet</span>
  </strong>
</p>

# 模型介绍

NowCastNet 是一个物理条件深度生成模型（physics-conditional deep generative model），用于极端降水的临近预报（nowcasting）。模型统一了物理演化方案（2D 连续性方程驱动的可微演化网络）与条件学习方法（SPADE 空间自适应归一化的生成网络），端到端优化预报误差，可生成物理合理、多尺度清晰的降水场外推，铅直预报时长达 3 小时。

论文：Skilful nowcasting of extreme precipitation with NowcastNet
https://www.nature.com/articles/s41586-023-06184-4

# 模型描述

- 演化网络（Evolution Network）：双路径 U-Net（共享演化 encoder + motion decoder + intensity decoder），实现可微演化算子（后向半拉格朗日平流），预测运动场与强度残差，输出 mesoscale 演化结果。
- 生成网络（Generative Network）：nowcast encoder + noise projector + nowcast decoder，通过 SPADE 空间自适应归一化以演化结果为条件，从潜变量采样生成对流尺度细节。
- 时间判别器（Temporal Discriminator）：多 kernel 3D 卷积 + 谱归一化，用于条件 GAN 训练与池化一致性正则。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 短临降水预报训练 | 使用雷达合成降水场（合成数据或 MRMS）训练演化网络与生成网络 |
| 短临降水预报推理 | 输入过去 9 帧雷达场，输出未来 20 帧降水场预测 |
| 评估与可视化 | 计算 CSI neighbourhood（16/32/64 mm/h）与 PSD 功率谱指标 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本 |

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

# 根据当前仓库自行设置
```bash
modelscope download --model OneScience/NowCastNet --local_dir ./model
cd model
```

### 安装运行环境


**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem、bio、all（全领域）
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai 
```

**GPU环境**
```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装，安装的时候注意自己对应的领域，目前支持earth、cfd、matchem（全领域）
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

Tier 1 复现使用合成雷达数据（按 2D 连续性方程平流+强度残差生成，200 样本，256×256，mm/h 单位，cap 128）。原始论文使用 MRMS 美国雷达（2016-2020 训练 / 2021 测试）与中国气象局（CMA）雷达；MRMS 需 NOAA 协议获取（https://www.nssl.noaa.gov/projects/mrms），中国数据需 CMA 许可。数据读取接口见 `model/synthetic_radar.py`（含 `SyntheticRadarDataset` 与 `MRMSDataset`，权限受限时自动回退到合成数据）。

### 训练

在此处提供运行脚本：

单卡：

```bash
# 阶段 1：演化网络
python scripts/train_evolution.py --config conf/evolution.yaml --out-dir checkpoints
# 阶段 2：生成网络（需先完成演化网络训练）
python scripts/train_generative.py --config conf/generative.yaml --out-dir checkpoints
```

训练会在 `checkpoints/` 下保存 `evolution_network.pt`、`generative_network.pt`、`temporal_discriminator.pt`。

### 训练权重

已提供训练权重（Tier 1 合成数据）：

- `weight/evolution_network.pt`（演化网络，17.7 MB）
- `weight/generative_network.pt`（生成网络，24.7 MB）
- `weight/temporal_discriminator.pt`（时间判别器，2.1 MB）

### 推理

```bash
python scripts/infer.py --config conf/generative.yaml \
  --evolution-checkpoint weight/evolution_network.pt \
  --generative-checkpoint weight/generative_network.pt \
  --input <input.npy> --output predictions.npy
```

推理会输出未来 20 帧降水场序列（N×20×H×W），保存至指定 `.npy` 文件。

### 评估和可视化

```bash
python scripts/evaluate.py --config conf/generative.yaml \
  --evolution-checkpoint weight/evolution_network.pt \
  --generative-checkpoint weight/generative_network.pt \
  --out-dir eval_out
```

评估会计算 CSI neighbourhood（16/32/64 mm/h）与 PSD 指标，写入 `eval_out/metrics.json`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

开源仓库复用上游内容，复现的论文可参考下面描述

- 本仓库为 NowcastNet 原始论文的复现版本（Tier 1 小数据端到端）。
- 论文：Zhang, Y. et al. Skilful nowcasting of extreme precipitation with NowcastNet. Nature 619, 526-532 (2023). DOI: 10.1038/s41586-023-06184-4.

