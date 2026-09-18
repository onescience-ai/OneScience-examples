<p align="center">
  <strong>
    <span style="font-size: 30px;">RFdiffusion</span>
  </strong>
</p>

# 模型介绍

RFdiffusion 是一种基于扩散模型的蛋白质骨架生成和设计方法，可用于无条件骨架生成、motif scaffolding、PPI/binder 设计和对称寡聚体采样。

# 模型描述

RFdiffusion 模型基于 RoseTTAFold 三轨网络与 SE(3) 等变去噪扩散过程的生成式蛋白质设计模型，可从随机结构逐步生成满足指定拓扑或功能约束的蛋白质骨架。

当前 ModelScope 包面向下载即用、本地快速验证和 OneCode 自动化运行场景，代码、配置、示例输入和权重均已放在当前目录内。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 无条件骨架生成 | 输入 contig 约束，输出设计骨架 PDB。 |
| Motif scaffolding | 输入包含 motif 的 PDB 和 contig 约束，输出 scaffold 设计结果。 |
| PPI/binder 设计 | 输入目标结构、hotspot 和 contig 参数，输出 binder 设计候选。 |
| 对称寡聚体采样 | 使用 symmetry 配置生成对称结构设计。 |
| ModelScope 全量包验证 | 使用包内 `config/ modules/ scripts/ examples/ weight/` 布局直接预检和推理。 |

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

### 2. 下载模型包并安装环境

```bash
modelscope download --model OneScience/RFdiffusion --local_dir ./RFdiffusion
```

### 训练权重
训练权重已经收录在weights文件夹内，可下载模型包后直接使用。

### 3. 运行预检

检查文件和真实权重：

```bash
python scripts/preflight.py --strict-weights
```

安装依赖后检查本地 import：

```bash
python scripts/preflight.py --strict-weights --strict-imports
```

只验证入口和 Hydra 配置，不执行采样：

```bash
RF_DIFFUSION_SMOKE_TEST=1 python scripts/run_inference.py
```

### 4. 运行推理

如果运行报错缺少.cache文件，可自行创建；

无条件骨架采样示例：

```bash
python scripts/run_inference.py \
  'contigmap.contigs=[80-80]' \
  diffuser.T=15 \
  inference.final_step=15 \
  inference.num_designs=1 \
  inference.write_trajectory=False \
  inference.output_prefix=outputs/smoke/design
```

Motif scaffolding 示例：

```bash
python scripts/run_inference.py \
  inference.input_pdb=examples/input_pdbs/1YCR.pdb \
  'contigmap.contigs=[10-40/A163-181/10-40]' \
  inference.output_prefix=outputs/motif/design
```

对称采样示例：

```bash
python scripts/run_inference.py --config-name symmetry \
  diffuser.T=15 \
  inference.final_step=15 \
  inference.output_prefix=outputs/symmetry/c2
```

### 5. 常用环境变量

```bash
export RF_DIFFUSION_MODEL_DIR=weight
export RF_DIFFUSION_INPUT_PDB=examples/input_pdbs/1qys.pdb
export RF_DIFFUSION_OUTPUT_PREFIX=outputs/design
export RF_DIFFUSION_SCHEDULE_DIR=.cache/schedules
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

RFdiffusion在BSD开源许可下发布（详见 [LICENSE](https://github.com/RosettaCommons/RFdiffusion/blob/main/LICENSE) 文件），可免费用于非营利和营利目的。
