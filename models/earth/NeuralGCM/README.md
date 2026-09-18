<p align="center">
  <strong>
    <span style="font-size: 30px;">NeuralGCM</span>
  </strong>
</p>

# 模型介绍

NeuralGCM（Neural General Circulation Models）是 Google Research 开源的混合机器学习与物理大气模型，用于天气预报和气候模拟。

论文：Neural General Circulation Models for Weather and Climate

https://arxiv.org/abs/2311.07222


# 模型描述

NeuralGCM 模型以可微分的大气动力学核心为基础，用神经网络表示未解析的物理过程、编码器和解码器，在保持物理约束的同时提高预报效率。

| Profile | 分辨率 | 类型 | 随包官方 checkpoint |
| :--- | :---: | :--- | :--- |
| `weather_forecast` | 0.7°（512 × 256） | 确定性天气预报，约 2 至 15 天 | `weight/models_v1_deterministic_0_7_deg.pkl` |
| `climate_scale` | 1.4°（256 × 128） | 确定性气候尺度模拟 | `weight/models_v1_deterministic_1_4_deg.pkl` |
| `forecast_2_8_deg` | 2.8°（128 × 64） | 确定性天气预报 | `weight/models_v1_deterministic_2_8_deg.pkl` |



# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气预报 | 使用 0.7° ERA5 数据训练短中期天气预报模型。 |
| 气候尺度模拟 | 使用 1.4° ERA5 数据进行较长时间的大气模拟模型训练。 |
| 低分辨率快速实验 | 使用 2.8° 低分辨率数据进行低成本的气象预报训练。 |
| 本地快速验证 | 用 `scripts/fake_data.py` 生成带正确通道协议的 HDF5 数据，检查数据、模型和 checkpoint 流程。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装 OneScience/JAX 依赖并运行脚本。 |
| 多卡训练 | 单机多卡并行训练。 |


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
modelscope download --model OneScience/NeuralGCM --local_dir ./NeuralGCM
cd NeuralGCM
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
modelscope download --dataset OneScience/ERA5 --local_dir ./data/era5
```

### 生成虚拟数据


```bash
python scripts/fake_data.py
```

脚本会在 `data/data/` 下生成各年份 HDF5 文件，在 `data/static.nc` 写入合成静态场，并在 `metadata/dataset_card.json` 保存通道、时间窗口和网格元数据。虚拟字段具有物理量纲近似值，但仅用于 shape、读取、重网格和数值稳定性验证。



### 训练

单卡：

```bash
# 0.7° 确定性短中期气象预报训练
python scripts/train_weather_forecast.py
# 1.4° 确定性气候尺度模拟训练
python scripts/train_climate_scale.py
# 2.8° 确定性低分辨率预报训练
python scripts/train_forecast_2_8_deg.py
```

多卡：

```bash
# 0.7° 确定性短中期气象预报训练
python scripts/train_weather_forecast.py --devices 8
# 1.4° 确定性气候尺度模拟训练
python scripts/train_climate_scale.py --devices 8
# 2.8° 确定性低分辨率预报训练
python scripts/train_forecast_2_8_deg.py --devices 8
```

# 微调

可选择从零训练保存的权重，或官方预训练权重对模型进行微调。

```bash
# 0.7° 确定性短中期气象预报模型微调
python scripts/train_weather_forecast.py --finetune ./data/checkpoint/model_bak.pkl
# 1.4° 确定性气候尺度模拟模型微调
python scripts/train_climate_scale.py --finetune ./data/checkpoint/model_bak.pkl
# 2.8° 确定性低分辨率预报模型微调
python scripts/train_forecast_2_8_deg.py --finetune ./data/checkpoint/model_bak.pkl
```

多卡微调时，同多卡从零训练相似，仅在运行指令后添加 `--devices` 参数即可。


### 训练权重

本项目已随附官方预训练 checkpoint：

| 本地文件 | 官方发布路径 |
| :--- | :--- |
| `weight/models_v1_deterministic_0_7_deg.pkl` | `gs://neuralgcm/models/v1/deterministic_0_7_deg.pkl` |
| `weight/models_v1_deterministic_1_4_deg.pkl` | `gs://neuralgcm/models/v1/deterministic_1_4_deg.pkl` |
| `weight/models_v1_deterministic_2_8_deg.pkl` | `gs://neuralgcm/models/v1/deterministic_2_8_deg.pkl` |
| `weight/models_v1_stochastic_1_4_deg.pkl` | `gs://neuralgcm/models/v1/stochastic_1_4_deg.pkl` |


### 推理


```bash
# 0.7° 确定性短中期气象预报推理
python scripts/inference.py --mode weather_forecast
# 1.4° 确定性气候尺度模拟推理
python scripts/inference.py --mode climate_scale.py
# 2.8° 确定性低分辨率预报推理
python scripts/inference.py --mode forecast_2_8_deg.py
```
默认使用训练得到的`./data/checkpoint/model_bak.pkl`作为推理模型，可通过`--checkpoint`参数修改推理所使用的权重。
默认输出为 `results/predictions.nc`，内容是官方命名的 pressure-level 变量和 rollout 时间坐标。

### 评估和可视化

```bash
python scripts/result.py
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 NeuralGCM 原始论文的复现版本。
- Google 发布的训练模型权重（包括本目录的四个 checkpoint）采用 Creative Commons Attribution-ShareAlike 4.0 International（CC BY-SA 4.0），与代码许可证不同。再分发或改编权重时必须遵守该协议并保留署名、相同方式共享等要求。
