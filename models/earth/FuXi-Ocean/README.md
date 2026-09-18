<p align="center">
  <strong><span style="font-size: 30px;">FuXi-Ocean</span></strong>
</p>

# 模型介绍

FuXi-Ocean 用于根据历史海洋状态和初始化大气变量进行 6 小时间隔的海洋场自回归预测，为高时空分辨率海洋预报方法研究提供可运行的工程验证流程。

论文：A deep learning global ocean forecasting model with sub-daily and eddy-resolving resolution  
https://doi.org/10.1038/s41612-026-01444-2

# 模型描述

该方法由天津大学、上海科学智能研究院、复旦大学、上海创新研究院、上海伏羲智算科技有限公司和海南热带海洋学院等机构的研究团队提出。论文使用 HYCOM 再分析与分析海洋场以及 ERA5 近地面大气变量开展训练。模型执行 6 小时间隔的温度、盐度、纬向流、经向流和海表高度自回归预测任务。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 海洋场自回归预测 | 在合成 sampled tiles 上验证由 4 帧历史海洋状态和初始化大气变量预测后续海洋状态的主任务。 |
| 核心方法验证 | 验证共享卷积编码、时空先验、历史特征融合、低分辨率 attention、共享解码和纬度加权 Charbonnier 损失。 |
| 本地工程验证 | 验证完整 `2160×4320` 科学网格的索引、重叠裁剪、数据契约、推理、评估和可视化接口；当前不生成完整全球预测。 |
| ModelScope/OneCode 运行 | 在 ModelScope 或 OneCode 环境中验证结构化数据、训练、推理、逐时效海洋预报指标和可视化流程。 |
| 多卡训练 | 通过 `torchrun` 验证分布式训练和 checkpoint 流程。 |

# 使用说明

## 1.OneCode

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2.下载安装

```bash
modelscope download --model OneScience/FuXi-Ocean --local_dir ./FuXi-Ocean
cd FuXi-Ocean
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

论文使用 HYCOM 海洋场和 ERA5 近地面大气变量，模型根据 4 个历史时刻预测 `2160×4320` 全球网格上的下一 6 小时海洋状态。本仓库使用 6 个结构化合成 tile，保留 105 个海洋通道、5 个大气通道、26 个深度层和完整全球索引协议。虚拟数据仅用于验证工程流程，不代表 HYCOM 或 ERA5 的真实数据分布、训练规模或论文性能。

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

训练使用纬度加权 Charbonnier 损失和多步自回归流程；默认配置仅缩小 tile、样本数、模型规模、训练轮数和 rollout 步数，不缩小真实变量与全局坐标协议。正式实验需要真实 HYCOM、ERA5 数据和完整计算资源，训练产物保存到：

```text
result/checkpoints/fuxi_ocean.pt
result/training/metrics.json
```

### 训练权重

本仓库不在 `weight/` 中内置官方权重，也未发现可确认的官方模型权重。论文代码可用性部分提供的完整资源链接为 https://doi.org/10.5281/zenodo.17412508 ，该链接不能视为已确认的权重下载链接。当前本地 checkpoint 是合成 `sampled_tiles` 的工程训练产物，不声明与论文或官方权重兼容。

### 推理

```bash
python scripts/inference.py
```

推理加载本地训练 checkpoint，默认在测试 tile 上执行 3 步自回归预测，同时记录论文的 40 步协议。结果明确标记为 `sampled_tiles`、`is_complete_global=false` 并保存覆盖率，不能视为已完成的全球预测；数值结果和元数据保存到：

```text
result/output/predictions.npz
result/output/metadata.json
```

### 评估和可视化

```bash
python scripts/result.py
```

评估在已采样 tile 上逐时效计算模型和 persistence 基线的纬度加权 RMSE 与 MBE，按 S、T、U、V 和 SSH 单位组报告并保留逐通道结果，不对单位不同的 105 个通道求统一均值。合成数据评估仅用于验证工程流程，不代表论文正式性能；结果保存到：

```text
result/evaluation/metrics.json
result/evaluation/lead_metrics.png
```

# OneScience官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

本仓库为 FuXi-Ocean 公开规格的独立工程复现版本。

本仓库代码的使用应遵循 Apache-2.0 许可证条款。

原论文按照 CC BY 4.0 许可证发布，引用和使用时应遵循该许可证条款。

HYCOM、ERA5、官方 Zenodo 产物和其他第三方数据或权重的使用应遵循各自项目的许可证及使用条款。
