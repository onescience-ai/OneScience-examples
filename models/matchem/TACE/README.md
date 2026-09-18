<p align="center">
  <strong>
    <span style="font-size: 30px;">TACE</span>
  </strong>
</p>

# 模型介绍

TACE（Tensor Atomic/Edge Cluster Expansion）是面向原子间势的等变模型，可预测能量、原子力和应力。

论文：*Tensor Atomic/Edge Cluster Expansion for Equivariant Many-Body Interatomic Potentials*
参考实现：https://github.com/xvzemin/tace

# 模型描述

TACE 基于 SO(3) 等变张量代数构建，使用等变消息传递架构对原子结构进行能量、受力和应力预测。模型支持从头训练、全参数/冻结参数/LoRA 微调，以及 ASE 可读结构的单点推理与评估。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 从头训练 | 使用 ASE 可读 XYZ 数据训练 TACE 模型 |
| 微调 | 基于预训练权重进行全参数、冻结参数或 LoRA 微调 |
| 单点推理 | 对 ASE 可读结构执行能量、受力和应力推理 |
| 分布式训练 | 支持单节点多 DCU 和多节点训练 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行训练。
- CPU 可以用于导入和小配置连通性验证，完整训练速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

### 下载模型包

```bash
modelscope download --model OneScience/TACE --local_dir ./tace
cd tace
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

本仓库不内置训练数据。以 BaTiO3 示例数据为例，配置文件中默认路径为 `${ONESCIENCE_DATASETS_DIR}/matchem/TACE/BaTiO3.xyz`。用户应提供与目标任务一致的 ASE 可读带标签轨迹，并在配置中将 `dataset.train_file` 指向该文件。

`demo/run.sh` 会自动将仓库根目录作为工作目录，输出保存在 `outputs/<name>_<timestamp>/`。

### 训练

使用 smoke 配置验证数据读取、建模、训练和 checkpoint 保存：

```bash
bash demo/run.sh --config demo/configs/tace_smoke_1dcu.yaml
```

使用完整配方从头训练：

```bash
bash demo/run.sh --config demo/configs/tace.yaml --submit
```

单节点 8 DCU 或两节点 16 DCU 训练：

```bash
bash demo/run.sh --config demo/configs/tace_8dcu.yaml --submit
bash demo/run.sh --config demo/configs/tace_2node_8dcu.yaml --submit
```

### 微调

为预训练权重生成微调配置：

```bash
python finetune.py --model weight/TACE-OAM-7M.pt
```

使用 DMC 数据的 smoke 微调测试：

```bash
bash demo/run.sh --config demo/configs/tace_finetune_dmc_smoke_1dcu.yaml --submit
```

微调模板（需替换为用户数据和模型路径后使用）：

```bash
bash demo/run.sh --config demo/configs/tace_finetune_1dcu.yaml --submit
```

### 推理

对 ASE 可读结构执行单点能量、受力和应力推理：

```bash
python single_point.py \
  --input structure.xyz \
  --model weight/TACE-OAM-7M.pt \
  --output predict.xyz \
  --device cuda \
  --batch_size 16
```

对带标签数据添加 `--test 1` 可输出误差指标。

### 训练权重

本仓库提供以下预训练权重：

| 文件 | 说明 |
| --- | --- |
| `weight/TACE-OAM-7M.pt` | OAM 预训练 TACE 模型，可用于单点推理或微调初始权重 |
| `weight/TACE-OMat24-7M.pt` | OMat24 预训练 TACE 模型 |
| `weight/TECE-OAM-RRA-1.0.pt` | OAM 预训练 TECE 模型 |
| `weight/TECE-OMat24-RRA-1.0.pt` | OMat24 预训练 TECE 模型 |

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |



# 引用与许可证

- TACE 相关代码来自 OneScience 项目中的 matchem 示例实现，并参考了上游 TACE 项目（https://github.com/xvzemin/tace）。上游 TACE 代码以 [MIT License](https://github.com/xvzemin/tace/blob/main/LICENSE) 发布。
- 如果在科研工作中使用 TACE 训练结果，建议引用 TACE 原始论文、OneScience 相关项目信息和实际使用的数据集来源。