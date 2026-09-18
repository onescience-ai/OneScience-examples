<p align="center">
  <strong>
    <span style="font-size: 30px;">MARIO</span>
  </strong>
</p>

# 模型介绍

MARIO 是由 ISAE-SUPAERO 相关团队提出的分辨率无关气动代理模型，可对不同几何构型和飞行工况下的速度场、压力场、湍流黏度场及表面压力分布进行快速预测。

本仓库基于 OneScience 技能，独立复现了 MARIO 论文中的 AirfRANS 流场预测实验。

论文：[MARIO: A Modulated Neural Field Framework for Resolution-Independent Surrogate Modeling of Aerodynamics](https://arxiv.org/abs/2505.14704)

# 模型描述

MARIO 基于调制条件神经场架构，使用 AirfRANS 二维翼型数据进行训练，面向跨网格分辨率的稳态气动流场预测。

# 适用场景

| 场景 | 说明 |
| ---------- | --------------------------------- |
| 二维翼型气动代理建模 | 面向 AirfRANS 风格二维翼型稳态 RANS 数据，预测速度、压力和湍流黏度 |
| 非结构网格场查询 | 训练时随机采样点，推理时可在完整 mesh 上分块查询 |
| 表面压力评估 | 额外统计 surface pressure MSE，用于分析近壁区域和气动力相关误差 |
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

```bash
modelscope download --model OneScience/MARIO --local_dir ./mario
cd mario
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**

```
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[cfd-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```


### 训练数据介绍

OneScience 社区提供可供训练的 airfrans 数据，用户可通过下述命令下载，并确认'conf/config.yaml'中数据路径设置正确.

```bash
modelscope download --dataset OneScience/airfrans --local_dir ./data
```
下载完成后，请修改 config/config.yaml 中的以下配置：
- paths.data_root：设置为 AirfRANS 数据集根目录，即包含 manifest.json 和各仿真样本的目录。
- paths.project_root：设置为当前 MARIO 项目的绝对路径。

### 训练

执行论文主实验的完整两阶段训练，包括 SDF 几何编码器训练和流场解码器训练：

```bash
python scripts/train.py --config config/config.yaml
```
训练过程中会通过标准输出实时打印几何编码损失、流场训练损失以及各物理场的 MSE。完整实验默认使用 AirfRANS scarce 任务的 200 个训练样本，并分别训练几何编码器和流场解码器 1000 个 epoch。

### 训练权重
仓库提供基于 AirfRANS 数据集训练得到的模型权重：`weight/best_model.pth`

### 推理和可视化

使用训练完成的 checkpoint 在 AirfRANS 完整测试网格上执行推理和评测：

```bash
python scripts/inference.py --config config/config.yaml
```
推理过程会逐样本实时输出 SDF 重建误差、各物理场归一化 MSE、表面压力 MSE、气动力系数及推理耗时。完整实验结果默认保存至 results/

完成推理后，可执行以下命令生成流场预测、真实值、绝对误差、表面压力以及论文精度对比图：
```bash
python scripts/result.py --config config/config.yaml
```


# 引用与许可证

- MARIO 原始论文：[MARIO: A Modulated Neural Field Framework for Resolution-Independent Surrogate Modeling of Aerodynamics](https://arxiv.org/abs/2505.14704)
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理；公开分发前请根据上游项目确认许可证要求。
