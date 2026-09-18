<p align="center">
  <strong>
    <span style="font-size: 30px;">SimpleFold</span>
  </strong>
</p>

# 模型介绍

SimpleFold 是 Apple 发布的生成式蛋白质折叠模型，可实现蛋白质结构预测，并设定训练、微调入口。

# 模型描述

SimpleFold 模型用通用 Transformer 层和 flow matching 目标从蛋白质 FASTA 序列预测三维结构。

当前 ModelScope 包内已包含默认 FASTA 示例、SimpleFold 权重、pLDDT 权重、CCD 辅助文件、Boltz 辅助权重和 ESM-2 3B 本地权重，下载完整模型包后可以直接使用。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 蛋白质结构预测 | 输入 FASTA，输出 mmCIF/PDB 结构。 |
| 本地离线推理整理 | 示例输入和权重已放入 `examples/`、`weight/`，脚本只使用本仓库文件。 |
| 训练/微调接口验证 | 用户按 `config/data/*.yaml` 准备 tokenized 数据后训练。 |
| ModelScope 标准包 | 使用 `config/ models/ scripts/ weight/` 布局。 |

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

想了解更多适配内容请联系 liubiao@sugon.com

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
#如果下述代码运行存在找不到库的情况，需要激活cuda，参考下列代码

```bash
source ${ROCM_PATH}/cuda/env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/fastpt/torch/lib:$LD_LIBRARY_PATH"
```
### 2. 下载模型包

```bash
modelscope download --model OneScience/SimpleFold --local_dir ./SimpleFold
cd SimpleFold
```

### 训练权重
训练权重已经包含在weights文件夹中，收纳simplefold-1B 100M、esm等多种权重

### 3. 运行推理

```bash
python scripts/run_inference.py \
  --simplefold_model simplefold_100M \
  --fasta_path examples/minimal.fasta \
  --output_dir outputs/minimal_inference \
  --num_steps 10 \
  --tau 0.01 \
  --nsample_per_protein 1 \
  --backend torch
```

输出目录：

```text
outputs/minimal_inference/predictions_simplefold_100M/
```

### 4. 训练——当前不提供对应数据集

训练前准备：

```text
datasets/
datasets/tokenized/
datasets/manifest.json
```

数据处理：

```bash
python scripts/process_data.py --data_dir /path/to/mmcif --out_dir datasets --num-processes 8
python scripts/tokenize_data.py --target_dir datasets --token_dir datasets/tokenized
```

`process_data.py` 默认使用包内 `weight/ccd.pkl`，无需额外下载 CCD 或启动 Redis；如需兼容旧 Redis CCD 流程，可显式传入 `--use-redis`。

训练：

```bash
python scripts/train.py
```

FSDP 训练：

```bash
python scripts/train_fsdp.py experiment=train_fsdp
```

微调/续训示例：

```bash
python scripts/train.py load_ckpt_path=weight/simplefold_100M.ckpt
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

SimpleFold 原始实现声明为 MIT License。本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理。科研使用请引用 SimpleFold 原始论文和 OneScience 相关项目信息。
