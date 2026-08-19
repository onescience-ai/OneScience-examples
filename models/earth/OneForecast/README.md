<p align="center">
  <strong>
    <span style="font-size: 30px;">OneForecast</span>
  </strong>
</p>

# 模型介绍

OneForecast 是由清华大学地球系统科学系黄小猛教授团队联合多家机构共同研发，是一个基于图神经网络（GNN）的全球-区域嵌套天气预报通用框架，核心目标是解决现有AI气象模型在平衡全球低分辨率与区域高分辨率预报、以及极端事件预报中存在的过度平滑等难题，相关论文已被 ICML 2025 接收。

论文: OneForecast: A Universal Framework for Global and Regional Weather Forecasting

https://arxiv.org/abs/2502.00338



# 仓库说明

本仓库是 OneScience 整理的 OneForecast 最小可运行独立模型仓库，面向 ModelScope 下载、OneCode 自动化运行和本地快速验证场景。

当前支持能力：

- 生成轻量级 ERA5 HDF5 测试数据。
- 单卡训练和 `torchrun` 分布式训练入口。
- 使用训练权重推理并保存预测结果。
- 绘制推理结果可视化图像。

当前不支持能力：

- 不随包提供真实 ERA5 数据或预训练权重。
- 默认配置 ERA5 虚拟数据生成，完整训练需要真实 ERA5 数据。
- 虚拟数据只用于流程连通性验证，不代表模型效果。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气预报训练 | 使用 ERA5 HDF5 数据训练单步 OneForecast 模型。 |
| 本地快速验证 | 使用虚拟数据检查数据协议、模型构建、训练、推理和结果可视化。 |
| 多卡训练 | 通过 PyTorch DDP 和 `torchrun` 在多张 GPU/DCU 上进行数据并行训练。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后安装 OneScience 依赖并运行。 |


# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `conf/config.yaml` | 训练、推理和数据配置 | 已适配本仓库相对路径 |
| `scripts/train.py` | 训练脚本 | 支持单卡和 torchrun 多卡 |
| `scripts/finetune.py` | 微调脚本 | 支持官方预训练权重或本项目训练权重微调 |
| `scripts/inference.py` | 推理脚本 | 需存在训练权重 |
| `scripts/result.py` | 评估与可视化脚本 | 读取 `result/output/*.npy` |
| `scripts/fake_data.py` | 假数据生成脚本 | 用于快速连通性验证 |
| `model/oneforecas.py` | 独立 Python 包 | oneforecas 的模型复现 |
| `model/era5_adapter.py` | 输入数据处理 | 将标准 ERA5 数据适配到 oneforecast 输入 |
| `weight/` | 权重目录 | 可放置预训练或发布权重 |



# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。



## 3. 快速开始


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



### 生成假数据进行流程验证

虚拟数据只用于检查数据协议和程序流程，不代表科学预报效果：

```bash
python scripts/fake_data.py
```

同时，OneScience 社区提供可供训练的 ERA5 数据（受数据文件大小限制，当前仓库内为完整数据切片），用户可通过下述命令下载，并确认 `conf/config.yaml` 中数据路径设置正确：

```bash
modelscope download --dataset OneScience/ERA5 --local_dir ./data
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
