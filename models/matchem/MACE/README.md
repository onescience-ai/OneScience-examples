<p align="center">
  <strong>
    <span style="font-size: 30px;">MACE</span>
  </strong>
</p>

# 模型介绍

MACE 是面向分子和材料体系的机器学习原子间势（MLIP）模型，基于 E(3)-等变图神经网络构建，可对原子结构进行能量和受力预测。

论文：*MACE: Higher order equivariant message passing neural networks for fast and accurate force fields*  
参考实现：https://github.com/ACEsuit/mace

# 模型描述

MACE 基于 E(3)-等变图神经网络架构，使用 HDF5/XYZ 格式数据进行训练，面向分子和材料体系开展能量与受力预测及结构优化。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 原子间势训练 | 使用标准配置读取 HDF5/XYZ 数据并训练 MACE 模型 |
| 分布式训练预检 | 检查多卡/多节点训练配置、数据路径和 statistics 是否一致 |
| 验证集评测 | 训练过程中在验证集上输出能量和力相关误差指标 |
| 自有数据迁移 | 参考现有配置替换为自有 HDF5/XYZ 数据和 statistics 文件 |
| 环境连通性验证 | 使用预检脚本检查 OneScience matchem 环境、pyyaml、h5py 和数据可读性 |

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
modelscope download --model OneScience/MACE --local_dir ./mace
cd mace
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

本仓库不内置训练数据。以 DMC 入门数据集为例，从 ModelScope 下载并放到仓库根目录的 `data/` 下：

```bash
modelscope download --dataset OneScience/DMC --local_dir ./data
```

下载后数据路径为 `data/data/DMC/`。`scripts/demo/run.sh` 会自动将仓库根目录作为 `ONESCIENCE_DATASETS_DIR`，因此无需手动设置该变量即可匹配配置文件中的路径。

其他配置（如 `ani1x_8dcu.yaml`、`water_*.yaml` 等）需要下载对应数据集并调整 YAML 中的数据路径。


### 训练

单卡：

```bash
bash scripts/demo/run.sh --config scripts/demo/configs/DMC.yaml
```

多卡：

```bash
# 以 8 卡为例，配置文件中 launch.launcher 应为 torchrun
bash scripts/demo/run.sh --config scripts/demo/configs/ani1x_8dcu.yaml
```

SLURM 提交：

```bash
bash scripts/demo/run.sh --config scripts/demo/configs/ani1x_8dcu.yaml --submit
```
### 训练权重

本仓库目前不内置训练权重，权重可通过上述训练得到。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |



# 引用与许可证

- MACE 相关代码来自 OneScience 项目中的 matchem 示例实现，并参考了上游 MACE 项目（https://github.com/ACEsuit/mace）。上游 MACE 代码以 [MIT License](https://github.com/ACEsuit/mace/blob/main/LICENSE) 发布。
- 如果在科研工作中使用 MACE 训练结果，建议引用 MACE 原始论文、OneScience 相关项目信息和实际使用的数据集来源。
