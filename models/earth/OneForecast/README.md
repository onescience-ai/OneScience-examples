<p align="center">
  <strong>
    <span style="font-size: 30px;">OneForecast</span>
  </strong>
</p>

# 模型介绍

OneForecast 是由清华大学地球系统科学系黄小猛教授团队联合多家机构共同研发，相关论文已被 ICML 2025 接收。

论文: OneForecast: A Universal Framework for Global and Regional Weather Forecasting

https://arxiv.org/abs/2502.00338

# 模型描述

OneForecast 是一个基于图神经网络（GNN）的全球-区域嵌套天气预报通用框架，核心目标是解决现有AI气象模型在平衡全球低分辨率与区域高分辨率预报、以及极端事件预报中存在的过度平滑等难题。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气预报训练 | 使用 ERA5 HDF5 数据训练单步 OneForecast 模型。 |
| 本地快速验证 | 使用虚拟数据检查数据协议、模型构建、训练、推理和结果可视化。 |
| 多卡训练 | 通过 PyTorch DDP 和 `torchrun` 在多张 GPU/DCU 上进行数据并行训练。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装 OneScience 依赖并运行。 |

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
modelscope download --model OneScience/OneForecast --local_dir ./OneForecast
cd OneForecast
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
modelscope download --dataset OneScience/ERA5 --local_dir ./data
```

### 生成虚拟数据

虚拟数据只用于检查数据协议和程序流程，不代表科学预报效果：

```bash
python scripts/fake_data.py
```


### 训练

单卡：

```bash
python scripts/train.py
```

训练默认从随机初始化开始，并在 `data/checkpoint/model_bak.tar` 保存模型。


多卡：

```bash
torchrun --nproc-per-node=4 scripts/train.py
```


### 微调

微调默认从训练阶段的 `data/checkpoint/model_bak.tar` 开始，并保存到`data/checkpoint/model_finetuned.tar`：

```bash
python scripts/finetune.py
```

### 训练权重

本仓库在 weights/ 文件夹内提供基于 ERA5 再分析数据训练的权重，权重文件即将上传，预计将于近期完成。


### 推理

推理默认加载训练检查点 `data/checkpoint/model_bak.tar`，使用测试年份数据，并将预测写入 `outputs/predictions/`：

```bash
python scripts/inference.py --config conf/config.yaml
```

### 结果可视化

```bash
python scripts/result.py
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 本仓库为 OneForecast 原始论文的复现版本。
