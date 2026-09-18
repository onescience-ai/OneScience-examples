<p align="center">
  <strong>
    <span style="font-size: 30px;">GraphCast</span>
  </strong>
</p>


# 模型介绍

GraphCast 是由 Google DeepMind 团队开发的全球中期天气预报模型，其核心论文发表于国际顶级期刊 Science

论文：GraphCast: Learning skillful medium-range global weather forecasting

https://arxiv.org/abs/2212.12794

# 模型描述
GraphCast 是基于图神经网络（GNN）构建的的全球中期天气预报模型，训练数据为 ECMWF 提供的 ERA5 全球大气再分析资料（1979–2017年）。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气预报研究 | 基于年度 ERA5 HDF5 数据训练 GraphCast 风格的图神经网络预报模型。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、辅助文件生成、训练入口和结果脚本。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |


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
modelscope download --model OneScience/GraphCast --local_dir ./GraphCast
cd GraphCast
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

### 生成辅助文件

```bash
python scripts/get_data_json.py
python scripts/compute_time_diff_std.py
```

生成文件：

- `data.json`
- `time_diff_std.npy`

### 训练

单卡：

```bash
python scripts/train.py
```

多卡：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练输出：

```text
data/checkpoints/model_bak.pth
data/checkpoints/trloss.npy
```

### 训练权重
本仓库在weights/文件夹内提供基于1979～2017年ERA5数据训练的权重，权重文件即将上传，预计将于近期完成。

### 微调

微调前需先完成训练并生成 `data/checkpoints/model_bak.pth`。

```bash
python scripts/finetune.py
```

微调输出：

```text
data/checkpoints/model_finetune_bak.pth
data/checkpoints/ft_trloss.npy
```


### 推理

推理默认读取 `data/checkpoints/model_finetune_bak.pth`：

```bash
python scripts/inference.py
```

预测结果输出到：

```text
result/output/
```

### 评估与可视化

```bash
python scripts/result.py
```

输出内容包括：

- `result/rmse.npy`
- `result/acc.npy`
- `result/loss.png`
- 指定日期和变量的预报对比图



# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Apache License 2.0。代码开源，允许商业及非商业使用
- 权重仅允许非商业用途

