<p align="center">
  <strong>
    <span style="font-size: 30px;">FourCastNet_v2</span>
  </strong>
</p>

# 模型介绍

FourCastNet v2 是 NVIDIA 及其合作团队提出的基于球面 Fourier Neural Operator（SFNO）的全球天气预报模型。

论文：Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere

https://arxiv.org/abs/2306.03838

# 模型描述

模型核心成果在于将架构从v1的自适应傅里叶神经算子（AFNO） 改为球面傅里叶神经算子（SFNO）。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气预报训练 | 使用 73 通道 ERA5 HDF5 数据训练 FourCastNet v2 风格 SFNO。 |
| 本地快速验证 | 使用合成 ERA5 文件检查训练、推理和结果可视化流程。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动 PyTorch DDP。 |

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
modelscope download --model OneScience/FourCastNet_v2 --local_dir ./FourCastNet_v2
cd FourCastNet_v2
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

OneScience 社区提供 ERA5 数据切片，可按配置下载：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

真实 HDF5 年度文件需要包含 `fields`、变量属性、`time_step`、`global_means` 和 `global_stds`。没有真实数据时，先生成仅用于流程验证的合成文件：

```bash
python scripts/fake_data.py
```

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 scripts/train.py
```
检查点默认保存到 `data/checkpoint/one_step/model_bak.pt`。

### 微调

单卡：

```bash
python scripts/train.py-stage finetune
```

多卡：

```bash
torchrun --nproc_per_node=8 scripts/train.py --stage finetune
```

检查点默认保存到 `data/checkpoint/finetune/model_bak.pt`。


### 训练权重

本仓库在 weights/ 文件夹内提供基于 ERA5 再分析数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

预测结果默认写入 `result/output/`。

### 评估和可视化

```bash
python scripts/result.py
```

默认生成纬度加权 RMSE/ACC 指标和 `result/figures/t2m_forecast.png`。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- FourCastNet v2 的 SFNO 数值实现沿用 NVIDIA Earth2MIP/相关官方实现的设计；上游代码和模型的许可证、版权声明必须保留。

