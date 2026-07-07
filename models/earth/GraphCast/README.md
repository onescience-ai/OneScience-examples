<p align="center">
  <strong>
    <span style="font-size: 30px;">GraphCast</span>
  </strong>
</p>


# 模型介绍

GraphCast 是 Google DeepMind 提出的基于图神经网络的全球中期天气预报模型。模型将经纬度规则网格编码到多尺度球面网格，在球面图上进行消息传递，再解码回规则网格，用于全球天气变量预报。

论文：GraphCast: Learning skillful medium-range global weather forecasting


# 仓库说明

当前支持能力：

- 生成轻量级 ERA5 HDF5 测试数据。
- 生成 GraphCast 损失函数所需的 `data.json` 和 `time_diff_std.npy`。
- 单卡训练和 `torchrun` 分布式训练入口。
- 基于训练权重继续自回归微调。
- 使用微调权重推理并保存 `.npy` 预测结果。
- 计算 RMSE/ACC，并绘制训练损失曲线和样例预报图。

当前不支持能力：

- 不随包提供真实 ERA5 数据或预训练权重。
- 默认配置面向 721 x 1440 的全球 0.25 度网格，完整训练需要较高显存和存储。
- 本独立包保留标准 DGL 图执行路径，不内置额外的图分区运行时；训练脚本可用 DDP 多进程复制模型训练。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球天气预报研究 | 基于年度 ERA5 HDF5 数据训练 GraphCast 风格的图神经网络预报模型。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、辅助文件生成、训练入口和结果脚本。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 集群训练 | 通过 `torchrun` 或 `scripts/work_slurm.sh` 启动多进程训练。 |


# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明 | 中文为主 |
| `configuration.json` | ModelScope/OneCode 元信息 | 标记 Pytorch 与通用任务 |
| `requirements.txt` | Python 依赖列表 | 执行 `pip install -r requirements.txt` |
| `config/config.yaml` | 模型、数据和训练配置 | 路径已适配独立包 |
| `scripts/_bootstrap.py` | 运行路径初始化 | 自动将 `model/` 加入 Python 路径并切换到包根目录 |
| `scripts/fake_data.py` | 生成轻量测试数据 | 输出到 `config/config.yaml` 中配置的数据目录 |
| `scripts/get_data_json.py` | 生成通道元数据 | 输出 `data.json` |
| `scripts/compute_time_diff_std.py` | 计算时间差标准差 | 输出 `time_diff_std.npy` |
| `scripts/train.py` | 训练入口 | 输出 `data/checkpoints/model_bak.pth` |
| `scripts/finetune.py` | 微调入口 | 依赖 `model_bak.pth` |
| `scripts/inference.py` | 推理入口 | 默认读取 `model_finetune_bak.pth` |
| `scripts/result.py` | 评估和可视化入口 | 输出 RMSE/ACC 和图片到 `result/` |
| `scripts/work_slurm.sh` | Slurm 示例脚本 | 提交前按集群修改资源参数 |
| `model/graphcast_src/` | 本地源码包 | 包含 GraphCast、ERA5 数据管道、图工具、GNN 模块和配置读取工具 |
| `weight/` | 权重占位目录 | 当前未内置权重 |

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- CPU 可以用于导入和小配置连通性验证，完整训练和推理速度较慢。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。

**软件要求**

请参考 `requirements.txt`。训练脚本会优先使用 NVIDIA Apex/环境内置 `apex.optimizers.FusedAdam`，若当前环境没有 Apex，则自动回退到 PyTorch `AdamW`。


**环境检测**

NVIDIA GPU：

```bash
nvidia-smi
```

海光 DCU：

```bash
hy-smi
```

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/GraphCast --local_dir ./GraphCast
cd GraphCast
```

### 安装运行环境

```bash
pip install -r requirements_dcu.txt
```

### 生成假数据进行流程验证

虚拟数据只用于流程连通性验证。若要快速跑通训练，请临时把 `config/config.yaml` 中的 `max_epoch`、`num_iters_step3`、`num_iters_step1`、`num_iters_step2` 等训练轮数改小。

```bash
python scripts/fake_data.py
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

单卡训练：

```bash
python scripts/train.py
```

多卡训练示例：

```bash
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=1000 --rdzv_backend=c10d --max_restarts=0 --master_addr="localhost" --master_port=29500 scripts/train.py
```

训练输出：

```text
data/checkpoints/model_bak.pth
data/checkpoints/trloss.npy
```

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

## 7. 评估与可视化

```bash
python scripts/result.py
```

输出内容包括：

- `result/rmse.npy`
- `result/acc.npy`
- `result/loss.png`
- 指定日期和变量的预报对比图


# 数据格式

真实数据存放在 `data/` 下，默认结构如下：

```text
data/
  data/
    1979.h5
    1980.h5
    ...
  static/
    land_mask.npy
    soil_type.npy
    topography.npy
```

年度 HDF5 文件需包含：

- `fields` 数据集，形状为 `[T, C, H, W]`
- `fields.attrs["variables"]`，变量名列表
- `fields.attrs["time_step"]`，时间间隔小时数
- `global_means`，形状为 `[1, C, 1, 1]`
- `global_stds`，形状为 `[1, C, 1, 1]`

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- GraphCast 原始论文：GraphCast: Learning skillful medium-range global weather forecasting。
- 如果在科研工作中使用本模型结果，建议引用 GraphCast 原始论文和实际使用的数据集。
