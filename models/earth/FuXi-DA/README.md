<p align="center"><strong><span style="font-size: 30px;">FuXi-DA</span></strong></p>

# 模型介绍

FuXi-DA 融合天气背景场和卫星观测生成改进的全球大气分析场，用于深度学习卫星数据同化和预报初值优化。模型在统一特征空间中学习背景误差、观测偏差和分析增量，可利用多时次静止卫星资料改善后续预报。

论文：FuXi-DA: a generalized deep learning data assimilation framework for assimilating satellite observations  
https://doi.org/10.1038/s41612-025-01039-3

# 模型描述

该方法由上海人工智能实验室、复旦大学等机构的研究团队提出。论文使用 ERA5、FuXi 背景场和风云四号B星 AGRI 观测开展训练。FuXi-DA 使用背景、观测和条件三条编码分支进行多尺度特征融合，并结合冻结 FuXi 的预报误差监督同化网络。模型适用于 0.25° 全球大气背景订正和静止卫星多通道观测同化。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 卫星数据同化 | 融合 70 通道背景场与 AGRI 多时次亮温观测。 |
| 多模态融合 | 验证背景、观测和条件分支的多尺度特征交互。 |
| 本地工程验证 | 在完整变量和全球坐标协议下执行采样 tile 验证。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证数据、训练、推理、同化指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/FuXi-DA --local_dir ./FuXi-DA
cd FuXi-DA
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

本仓库使用少量虚拟样本验证工程流程，虚拟数据包含 70 变量全球背景场、8 个 AGRI 时次、15 个观测通道、6 小时时间关系，以及 `[70,721,1440]` 背景和 `[8,15,640,640]` 观测的真实维度协议。虚拟数据通过程序化完整坐标场和少量原坐标对齐 tile 保持变量、通道和时空规格，仅减少实际执行的样本与覆盖范围。该数据仅用于验证多分支 U-Net、特征融合、训练、推理和评估流程，不代表 ERA5 或 AGRI 官方数据分布与训练规模。

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

训练联合使用纬度加权分析 L1 和冻结预报代理提供的多时效监督，使分析场同时接近 ERA5 并改善后续预报。默认配置缩小实际 tile 覆盖、网络宽度、迭代次数和预报监督步数，论文配置保留 6000 次迭代和 10 个预报时效。训练产物保存到：

```text
result/checkpoints/fuxi_da.pt
result/training/metrics.json
```

### 训练权重

论文未提供可直接加载的官方模型权重，本仓库不在 `weight/` 中内置权重。

### 推理

```bash
python scripts/inference.py
```

推理加载训练 checkpoint，以 70 通道背景 tile 和对应 AGRI 多时次观测作为输入。模型通过背景、观测和条件分支生成分析增量，并将其叠加到背景场得到分析结果。输出保留目标、背景、观测、分析、坐标和覆盖信息，并明确标记为非完整全球覆盖。推理结果保存到：

```text
result/output/predictions.npz
```

### 评估和可视化

```bash
python scripts/result.py
```

评估比较原始背景、纯订正基线和 FuXi-DA 分析的纬度加权 RMSE，并保存 Z、T、U、V、R 和地表变量组结果。评估还计算逐预报步误差及单观测扰动产生的分析增量局地性，并生成不同分析方案和变量组误差对比图。虚拟 tile 结果仅用于工程验证，不代表论文完整全球性能。评估结果保存到：

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

本仓库为 FuXi-DA 公开规格的独立工程复现版本，代码采用 Apache License 2.0 许可证。

原始论文采用 CC BY-NC-ND 4.0 许可证；论文、官方模型权重、ERA5 和风云四号B星 AGRI 数据仍应按照各自项目的许可证及使用条款使用。
