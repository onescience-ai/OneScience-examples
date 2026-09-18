<p align="center">
  <strong>
    <span style="font-size: 30px;">FuXi-S2S</span>
  </strong>
</p>

# 模型介绍

FuXi-S2S 是由复旦大学等机构提出的全球次季节预报模型。

论文：A machine learning model that outperforms conventional global subseasonal forecast models

https://doi.org/10.1038/s41467-024-50714-1


# 模型描述

FuXi-S2S 接收两个连续日平均大气状态，面向传统数值模式较具挑战性的两周至两个月预报范围。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球次季节预报 | 使用官方 FuXi-S2S ONNX 权重和符合固定 76 通道顺序的 ERA5 输入进行推理。 |
| 本地快速验证 | 使用合成 HDF5 数据检查、数据加载、ONNX Runtime 和可视化流程。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后配置 ONNX Runtime provider 并运行。 |

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
modelscope download --model OneScience/FuXi-S2S --local_dir ./FuXi-S2S
cd FuXi-S2S
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

官方模型使用固定顺序的 76 通道 ERA5 日平均状态。OneScience 社区提供 ERA5 数据切片：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

没有真实数据时，可生成仅用于接口检查的原生网格合成数据：

```bash
python scripts/fake_data.py
```


### 推理

默认配置面向 DCU；请先确认 conf/config.yaml 中的 device 和 providers 与运行环境一致：

```bash
python scripts/inference.py
```

各 ONNX 输出默认保存到 result/output/。

### 评估和可视化

```bash
python scripts/result.py
```

脚本默认读取 result/output/ 中最新的 NPY 输出，并在 result/visualization/ 生成多变量全球预报图。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 FuXi-S2S 官方项目的 DCU 适配版本。
- 本模型包包含的官方 ONNX 权重及外部数据文件受 Zenodo 发布页所列 CC BY-NC-ND 4.0 条款约束。
