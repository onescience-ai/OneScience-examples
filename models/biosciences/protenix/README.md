<p align="center">
  <strong>
    <span style="font-size: 30px;">Protenix</span>
  </strong>
</p>

# 模型介绍

Protenix 是面向蛋白质、核酸和配体等生物分子复合物结构预测的 AlphaFold3-like 模型，可从 JSON 描述的分子输入和本地 MSA 特征预测三维结构，并输出 CIF 结构文件与置信度结果。

# 模型描述

Protenix 模型采用 Pairformer 表征网络与原子级扩散 Transformer，可统一预测蛋白质、核酸、配体及其复合物结构。

当前 ModelScope 包面向下载即用、本地快速验证和 OneCode 自动化运行场景，代码、配置、示例输入和预训练权重均已放在当前目录内。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 复合物结构预测 | 输入 Protenix JSON 和本地 MSA，输出预测 CIF 与置信度 JSON。 |
| ModelScope 全量包验证 | 使用包内 `config/ models/ scripts/ examples/ weight/` 布局直接预检和推理。 |
| 微调链路验证 | 使用包内权重和 `ft_datasets/finetune_subset.txt` 启动微调入口。 |
| 训练接口整理 | 用户提供完整 Protenix 数据集后，可运行单卡训练脚本。 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用GPU或DCU运行。
- CPU可以用于连通性验证，但速度较慢。
- DCU用户需要预先安装DTK，建议使用DTK 25.04.2以上版本或与当前集群匹配的OneScience推荐版本。

**软件要求**

DCU用户想了解更多适配内容请联系 liubiao@sugon.com

**环境检测**

- NVIDIA GPU：

```bash
nvidia-smi
```

- 海光DCU：

```bash
hy-smi
```

## 快速开始

### 1. 安装运行环境

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[bio] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

### 2. 下载模型包&数据集

```bash
modelscope download --model OneScience/protenix --local_dir ./protenix
modelscope download --dataset OneScience/protenix_dataset --local_dir ./protenix_dataset
cd protenix
```
### 训练权重

训练权重已包含在weights文件夹内，下载模型包后可直接使用。

### 3. 运行预检

仅检查模型包、权重和本地导入：

```bash
python scripts/preflight.py --strict-weights --strict-imports
```

如果已经准备完整数据集：

```bash
export DATA_ROOT_DIR=../protenix_dataset
python scripts/preflight.py --strict-weights --strict-imports --strict-data
```

### 4. 运行推理

```bash
export DATA_ROOT_DIR=../protenix_dataset
bash scripts/inference_unified_demo.sh
```

默认输出目录：

```text
output_unified/7r6r/seed_101/predictions/
```

### 5. 训练与微调

训练：

```bash
export DATA_ROOT_DIR=../protenix_dataset
bash scripts/train_demo.sh
```

微调：

```bash
export DATA_ROOT_DIR=../protenix_dataset
bash scripts/finetune_demo.sh
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

Protenix 项目，包括代码和模型参数，依据 [Apache 2.0 许可协议](https://github.com/bytedance/Protenix/blob/main/LICENSE) 提供，可免费用于学术研究和商业用途。
