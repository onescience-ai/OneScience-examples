<p align="center">
  <strong>
    <span style="font-size: 30px;">AIFS_Single_v1</span>
  </strong>
</p>

# 模型介绍

AIFS Single v1.1 是由欧洲中期天气预报中心（ECMWF） 开发的人工智能天气预报系统确定性版本。

论文：AIFS — ECMWF's data-driven forecasting system, arXiv:2406.01465

https://arxiv.org/abs/2406.01465

# 模型描述

AIFS 是基于图神经网络（GNN）构建完成，通过 EAR5 数据预训练以及 NWP 业务分析数据微调得到。



# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 天气预报训练 | 使用 ERA5 HDF5 数据从零训练 AIFS |
| 本地快速验证 | 使用虚拟数据检查数据读取、模型训练与推理、推理结果可视化。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |



# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本。

### 下载模型包

```bash
modelscope download --model OneScience/AIFS_Single_v1 --local_dir ./AIFS_Single_v1
cd AIFS_Single_v1
```

### 安装运行环境


**DCU**
```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[earth-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU**
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


### 训练

```bash
python scripts/train.py
```

训练权重保存至 `weights/model_bak.ckpt`，训练前计算得到的归一化文件保存至`weights/era5_stats.npz`

### 训练权重
本仓库在weights/文件夹内提供基于ERA5再分析数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

预报步数通过 `conf/config.yaml` 中 `test_lead_time` 控制（小时数，默认 24 = 1 天）。
推理结果将保存至 `output`目录


### 评估和可视化

```bash
python scripts/result.py
```

计算 ACC / RMSE 指标并绘图。指标保存至 `metrics/`，图片保存至 `plots/`。


# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 AIFS Single v1.1 原始论文的复现版本。

