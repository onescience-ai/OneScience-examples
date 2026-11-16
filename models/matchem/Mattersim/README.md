<p align="center">
  <strong>
    <span style="font-size: 30px;">MatterSim</span>
  </strong>
</p>

# 模型介绍

MatterSim 是微软研究院提出的跨元素、温度和压力的深度学习原子间势模型，可对无机材料、分子及周期性体系进行能量和受力预测。

论文：*MatterSim: A deep-learning atomistic model across elements, temperatures, and pressures*  
参考实现：https://github.com/microsoft/mattersim

# 模型描述

MatterSim 基于深度学习架构，使用多种材料与分子数据集进行训练，面向无机材料、分子及周期性体系开展能量与受力预测、结构弛豫、分子动力学及自定义数据集微调。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 单点能量/受力预测 | 对给定原子结构快速预测能量和原子受力 |
| 批量结构推理 | 对多个结构进行批量能量/受力预测 |
| 结构弛豫 | 使用 FIRE/BFGS 优化原子位置和晶胞形状 |
| 分子动力学 | NVT 系综下运行短程 MD 采样 |
| 自定义数据微调 | 在自有数据集上微调预训练 MatterSim 模型 |
| 环境连通性验证 | 使用单点/弛豫脚本检查 OneScience matchem 环境、模型加载和 CUDA/DCU 可用性 |

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
modelscope download --model OneScience/Mattersim --local_dir ./mattersim
cd mattersim
```

### 安装运行环境

**DCU环境**

```bash
# 请首先激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

**GPU环境**

```bash
# 请首先激活CONDA
conda create -n onescience311 python=3.11 -y libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
# 支持uv安装
pip install onescience[matchem-gpu] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```


### 训练数据介绍

本仓库默认仅内置 `high_level_water.xyz` 示例数据，用于快速验证模型加载、单点推理、结构弛豫与分子动力学与微调流程。如需其他训练数据，请自行下载并放到 `data/` 目录。

### 训练权重

仓库已内置 `weight/mattersim-v1.0.0-1M.pth`，同时脚本均支持通过 `--checkpoint` 指定模型权重。

### 推理

```bash
cd scripts
python single_point.py --checkpoint ../weight/mattersim-v1.0.0-1M.pth
```

```bash
cd scripts
python batch_inference.py --checkpoint ../weight/mattersim-v1.0.0-1M.pth
```

**结构弛豫**

```bash
cd scripts
python relax.py --checkpoint ../weight/mattersim-v1.0.0-1M.pth --device cuda
```

> 默认使用 `../weight/mattersim-v1.0.0-1M.pth`。

**分子动力学**

```bash
cd scripts
python md.py --checkpoint ../weight/mattersim-v1.0.0-1M.pth --device cuda
```

> 同样默认使用 `../weight/mattersim-v1.0.0-1M.pth`。

### 微调

直接修改 `scripts/finetune_config.yaml` 中的路径和参数（例如 `train_data_path`、`checkpoint` 等）：

```bash
cd scripts
# 编辑 finetune_config.yaml 中的 train_data_path、checkpoint 等字段
```

单卡：

```bash
python finetune.py --config finetune_config.yaml
```

多卡 DDP：

```bash
torchrun --nproc_per_node=4 finetune.py --config finetune_config.yaml
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

---

# 引用与许可证

- MatterSim 相关代码来自 OneScience 项目中的 matchem 示例实现，并参考了上游 MatterSim 项目（https://github.com/microsoft/mattersim）。上游 MatterSim 代码以 [MIT License](https://github.com/microsoft/mattersim/blob/main/LICENSE) 发布。
- 如果在科研工作中使用 MatterSim 训练或推理结果，建议引用 MatterSim 原始论文、OneScience 相关项目信息和实际使用的数据集来源。

