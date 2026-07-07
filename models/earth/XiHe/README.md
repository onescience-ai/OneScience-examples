<p align="center">
  <strong>
    <span style="font-size: 30px;">Xihe</span>
  </strong>
</p>


# 模型介绍

Xihe（羲和）是面向高分辨率全球海洋预报的 Transformer 模型，用于学习海表高度、海表温度、10m 风场以及多层海温、盐度和流速变量之间的时空演化关系，适用于全球海洋涡解析预报研究。

论文：XiHe: A Data-Driven Model for Global Ocean Eddy-Resolving Forecasting


# 仓库说明

当前支持能力：

- 生成 CMEMS 风格 HDF5 测试数据、统计文件和海陆掩码。
- 单卡训练和 `torchrun` 分布式训练入口。
- 使用训练权重推理并保存 `.npy` 预测结果。
- 计算 RMSE/ACC，并绘制训练/验证损失曲线和样例预报图。

当前不支持能力：

- 不随包提供真实 CMEMS 数据或预训练权重。
- 默认配置面向 2041 x 4320 的全球高分辨率海洋网格，完整训练需要较高显存和存储。
- 虚拟数据只用于流程连通性验证，不代表模型效果。


# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 全球海洋预报研究 | 基于年度 CMEMS HDF5 数据训练 Xihe 风格的海洋预报模型。 |
| 本地快速验证 | 使用虚拟数据检查数据读取、训练入口、推理和结果脚本。 |
| ModelScope/OneCode 运行 | 作为独立模型包下载后直接安装依赖并运行脚本。 |
| 多卡训练 | 通过 `torchrun`启动多进程训练。 |


# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明 | 中文为主 |
| `configuration.json` | ModelScope/OneCode 元信息 | 标记 Pytorch 与通用任务 |
| `requirements.txt` | Python 依赖列表 | 标准安装入口 |
| `requirements_dcu.txt` | DCU 环境依赖列表 | 当前与 `requirements.txt` 保持一致，可按集群实际修改 |
| `config/config.yaml` | 模型、数据和训练配置 | 路径已适配独立包 |
| `scripts/_bootstrap.py` | 运行路径初始化 | 自动将 `model/` 加入 Python 路径并切换到包根目录 |
| `scripts/fake_data.py` | 生成轻量测试数据 | 输出到 `config/config.yaml` 中配置的数据目录 |
| `scripts/train.py` | 训练入口 | 输出 `data/checkpoints/model_bak.pth` |
| `scripts/inference.py` | 推理入口 | 默认读取 `model_bak.pth` |
| `scripts/result.py` | 评估和可视化入口 | 输出 RMSE/ACC 和图片到 `result/` |
| `scripts/work_slurm.sh` | Slurm 示例脚本 | 提交前按集群修改资源参数 |
| `model/xihe_src/` | 本地源码包 | 包含 Xihe、CMEMS 数据管道、SIE/Attention/MLP 模块和配置读取工具 |
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

请参考 `requirements_dcu.txt`。标准 Python 环境也可以使用：

```bash
pip install -r requirements_dcu.txt
```

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
modelscope download --model OneScience/Xihe --local_dir ./Xihe
cd Xihe
```

### 安装运行环境

```bash
pip install -r requirements_dcu.txt
```

### 生成假数据进行流程验证

虚拟数据只用于流程连通性验证。默认配置保持论文级高分辨率尺寸；若要快速跑通训练，请临时把 `config/config.yaml` 中的 `img_size`、`in_chans`、`out_chans`、`channels` 和 `max_epoch` 等参数调小，再运行假数据脚本。

```bash
python scripts/fake_data.py
```

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
data/checkpoints/valoss.npy
```

### 推理

推理默认读取 `data/checkpoints/model_bak.pth`：

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
    2010.h5
    2013.h5
    2014.h5
  stats/
    global_means.npy
    global_stds.npy
  land_mask.npy
  metadata.json
```

年度 HDF5 文件需包含：

- `fields` 数据集，形状为 `[T, C, H, W]`
- `fields.attrs["variables"]`，变量名列表
- `fields.attrs["time_step"]`，时间间隔小时数
- `global_means.npy`，形状为 `[1, C, 1, 1]`
- `global_stds.npy`，形状为 `[1, C, 1, 1]`
- `land_mask.npy`，形状为 `[H, W]`

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Xihe 原始论文：XiHe: A Data-Driven Model for Global Ocean Eddy-Resolving Forecasting。
- 如果在科研工作中使用本模型结果，建议引用 Xihe 原始论文和实际使用的数据集。
