<p align="center">
  <strong><span style="font-size: 30px;">Geoi-DBN</span></strong>
</p>

# 模型介绍

Geoi-DBN 融合卫星、地面站和气象数据估算中国陆地区域的日尺度地面 PM2.5 浓度，主要用于空气污染空间制图和多源观测融合研究。

论文：Estimating Ground-Level PM2.5 by Fusing Satellite and Station Observations: A Geo-Intelligent Deep Learning Approach  
https://doi.org/10.1002/2017GL075710

# 模型描述

该方法由武汉大学团队提出。论文融合了 2015 年中国环境监测总站（CNEMC）地面观测、MODIS 气溶胶光学厚度、MERRA-2 气象资料和 MODIS NDVI 数据。模型执行日尺度地面 PM2.5 浓度的融合估算。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| PM2.5 回归 | 输入 10 维气溶胶、气象、植被、时空邻域和污染源距离特征，输出日尺度 PM2.5 浓度。 |
| 地理特征与 DBN 方法验证 | 验证训练折安全的空间/时间 PM2.5 特征、两层 RBM 逐层预训练和 DBN 回归微调。 |
| 本地工程验证 | 验证结构化数据生成、特征构建、训练、checkpoint、推理和回归评估流程。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证 Geoi-DBN 的完整工程流程。 |
| 多卡训练 | 通过 `torchrun` 验证标准分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/Geoi-DBN --local_dir ./Geoi-DBN
cd Geoi-DBN
```

### 环境依赖

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可用于默认小样本配置的连通性验证。
- DCU 用户需预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**DCU环境**

```bash
# 请首先激活 DTK 及 Conda
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活 Conda
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[earth-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

模型输入为 `AOD, RH, WS, TMP, PBL, PS, NDVI, S_PM25, T_PM25, DIS` 共 10 维特征，目标为地面 PM2.5 浓度。仓库样例遵循 2015 年中国 `0.1 degree` grid-day 坐标协议，仅采样少量连续网格和日期。虚拟数据仅用于验证工程流程，不代表 CNEMC、MODIS、MERRA-2 或 NDVI 的真实数据分布、论文训练规模和正式性能。

```bash
python scripts/fake_data.py
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

训练先对 `10 -> 15 -> 15 -> 1` 网络执行两层 RBM 的 CD-k 逐层预训练，再使用 MSE 监督微调，配置位于 `conf/config.yaml`。默认配置面向合成小样本工程验证，正式实验需要真实观测数据与相应计算资源，训练产物保存到：

```text
result/checkpoints/geoi_dbn.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置官方权重，也未发现可确认的官方 checkpoint。当前 `result/checkpoints/geoi_dbn.pt` 是运行本仓库训练脚本后生成的本地工程产物，不声明与论文或官方权重兼容。

### 推理

```bash
python scripts/inference.py
```

推理加载本地训练 checkpoint，为全部样本构建仅引用训练折标签且排除查询样本自身的地理特征，并输出 `[N,1]` 日尺度 PM2.5 预测。结果包含观测值、物理输入、坐标、日期、数据划分和合成数据标记，保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估计算 PM2.5 回归指标和空间分组结果，并生成观测、预测及误差对比图。虚拟数据结果仅用于验证工程流程，不代表论文正式性能；结果保存到：

```text
result/evaluation/metrics.json
result/evaluation/comparison.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 Geoi-DBN 公开规格的独立工程复现版本。

本仓库代码、官方模型权重和数据的使用仍应以各自项目中的许可证及使用条款为准。
