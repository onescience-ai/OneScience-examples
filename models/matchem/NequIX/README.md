<p align="center">
  <strong>
    <span style="font-size: 30px;">NequIX</span>
  </strong>
</p>

# 模型介绍

NequIX 是面向材料体系的机器学习原子间势（MLIP）模型，基于 E(3)-等变图神经网络构建，可对原子结构进行能量、原子力和应力预测。上游 NequIX 以 JAX 为主、附带 PyTorch 后端；本仓库集成并发布的是其 **PyTorch 后端**（JAX-free），可在 DCU/ROCm 上运行（`kernel=false`，OpenEquivariance kernel 仅支持 CUDA）。

论文：*Training a foundation model for materials on a budget*（arXiv:2508.16067）
参考实现：https://github.com/atomicarchitects/nequix

# 模型描述

本仓库提供 OneScience 集成的 NequIX torch 后端：`model/` 是模型架构源码镜像（`NequixTorch`、JAX-free 的 `.pt` 加载器、DDP 采样器、e3nn 0.4.4 兼容补丁），`weight/nequix-mp-1.pt` 是 `nequix-mp-1` 预训练模型（由官方 `nequix-mp-1.nqx` 一次性转换得到），`train.py`/`single_point.py` 是训练与推理入口，`demo/` 提供 Slurm/DCU 训练配置。

`nequix-mp-1` 在 MPtrj（Materials Project 2023，DFT PBE+U）上训练，覆盖 89 种元素，cutoff 6.0 Å。

# 适用场景

| 场景 | 说明 |
| :---: | --- |
| 单点能量与受力推理 | 使用 `nequix-mp-1.pt` 预测结构的能量、原子力和应力 |
| 从头训练 | 使用 MPtrj `.aselmdb` 数据从头训练 NequIX 模型 |
| 微调 | 以 `nequix-mp-1.pt` 为起点，在自有 `.aselmdb` 数据上微调 |
| 分布式训练 | 使用 `torchrun` 多卡/多 DCU DDP 训练 |
| Slurm/DCU 训练 | 使用仓库配置和启动脚本提交单卡或多卡作业 |

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
modelscope download --model OneScience/NequIX --local_dir ./nequix
cd nequix
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

本仓库不内置训练数据。MPtrj 数据集单独发布于 `OneScience/Mptrj`，从 ModelScope 下载并放到仓库根目录的 `data/` 下：

```bash
modelscope download --dataset OneScience/Mptrj --local_dir ./data
```

下载后数据路径为 `data/data/Mptrj/`，包含 `aselmdb_smoke`（smoke 训练）、`aselmdb_mptrj`（全量 train/val）和 `hdf5_filtered`（HDF5 格式 train/val/test）三个子集。`demo/run.sh` 会把仓库根目录作为 `ONESCIENCE_DATASETS_DIR`，因此无需手动设置该变量即可匹配配置文件中的路径。

### 训练

smoke 训练（单卡，`aselmdb_smoke` 数据，1 个 epoch）：

```bash
bash demo/run.sh --config configs/mptrj_smoke.yaml
# 或提交到 Slurm
bash demo/run.sh --config configs/mptrj_smoke.yaml --submit
```

2 DCU DDP 训练（`n_dcu: 2`，每卡 `batch_size: 1`）：

```bash
bash demo/run.sh --config configs/mptrj_2dcu_smoke.yaml --submit
```

mp-1 从头训练（`aselmdb_mptrj` 全量数据）：

```bash
bash demo/run.sh --config configs/mptrj_full.yaml --submit
```

多卡/多 DCU 训练：在 YAML 中设置 `n_dcu`，`run.sh` 会用 `torchrun` 启动（每卡一个进程）。`batch_size` 是每卡的 batch size。训练输出保存在 `outputs/<name>_<timestamp>/`，包括配置快照、`checkpoint.pt`（最优 EMA 模型）和 `state.pkl`（可续训状态）。

### 推理

推理脚本默认加载本仓库的 `weight/nequix-mp-1.pt`。单点计算默认使用周期性 bulk Cu：

```bash
python single_point.py
```

读取 ASE 支持的结构文件：

```bash
python single_point.py \
  --input structure.cif \
  --output outputs/single_point.json
```

指定其他兼容的 torch checkpoint：

```bash
python single_point.py --model /path/to/model.pt --input structure.cif
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- NequIX 相关代码来自 OneScience 项目中的 MatChem 集成，并参考了上游 NequIX 项目（https://github.com/atomicarchitects/nequix，commit `da0fb24`）。上游 NequIX 代码以 [MIT License](https://github.com/atomicarchitects/nequix/blob/main/LICENSE) 发布。
- `nequix-mp-1` 模型权重由官方 `nequix-mp-1.nqx`（JAX 格式）一次性转换得到，其再分发权限以上游 NequIX 发布条款为准。
- 训练数据 MPtrj 单独发布于 [OneScience/Mptrj](https://modelscope.cn/datasets/OneScience/Mptrj)，其许可和来源信息以该数据集卡片及 Materials Project 原始来源说明为准。
- 如果在科研工作中使用 NequIX 或 `nequix-mp-1` 训练结果，建议引用 NequIX 原始论文、OneScience 相关项目信息和实际使用的数据集来源。
