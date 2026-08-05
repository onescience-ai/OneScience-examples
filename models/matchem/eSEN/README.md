<p align="center">
  <strong>
    <span style="font-size: 30px;">eSEN</span>
  </strong>
</p>

# 模型介绍

eSEN（equivariant Smooth Energy Network）是 FAIR Chemistry 提出的等变图神经网络原子间势，可预测材料结构的能量、原子力和应力。OneScience 提供 ASE calculator 和微调入口，可用于单点计算、结构弛豫、分子动力学以及自定义材料数据微调。

上游实现：[FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem)

# 模型描述

eSEN 以周期原子结构图为输入，通过旋转等变表示学习平滑的势能面，并由能量梯度获得守恒力。不同预训练 checkpoint 对应不同材料数据域，使用时应选择与目标元素体系及 DFT 标注设置接近的权重。

本示例包含 eSEN 模型代码、推理脚本、oxide PBE 微调示例和多卡启动配置。受限的 eSEN 预训练 checkpoint 不随仓库发布。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 单点计算 | 预测周期结构的总能量、原子力和应力 |
| 结构弛豫 | 使用 ASE BFGS 优化原子位置和可选晶胞 |
| 分子动力学 | 使用 ASE Langevin 运行 NVT 轨迹 |
| Oxide PBE 微调 | 使用 energy 和 forces 标签微调 MPTrj checkpoint |
| 分布式微调 | 支持单机多卡和 Slurm 多节点 DDP |

本仓库不提供从随机初始化开始的完整预训练流程，训练入口用于 checkpoint 微调。

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 进行推理和微调。
- CPU 可用于导入和配置检查，不建议用于正式计算。
- DCU 需要加载与 PyTorch 构建匹配的 DTK 运行时。

### 获取运行资源

示例代码已位于当前目录。下载 ModelScope 模型包以获取旋转基文件 `Jd.pt`：

```bash
cd models/matchem/eSEN
modelscope download --model OneScience/eSEN --local_dir ./resources/eSEN
mkdir -p weight
cp resources/eSEN/weight/Jd.pt weight/Jd.pt
```

### 安装运行环境

**DCU 环境**

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[matchem-dcu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

**GPU 环境**

```bash
conda create -n onescience311 python=3.11 -y \
  libstdcxx-ng=12 libgcc-ng=12 gcc_linux-64=12 gxx_linux-64=12
conda activate onescience311
pip install onescience[matchem-gpu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

### 预训练权重

eSEN 预训练 checkpoint 需要在 FAIR Chemistry 官方模型发布页申请访问权限，本仓库不提供或重新分发这些权重。获得权限后，将需要的 checkpoint 放到：

```text
weight/
├── Jd.pt                       # 从 OneScience/eSEN 模型包下载的旋转基文件
├── esen_30m_mptrj.pt            # 用户申请后自行放置
├── esen_30m_omat.pt             # 用户申请后自行放置
└── esen_30m_oam.pt              # 用户申请后自行放置
```

权重说明：

| 权重 | 训练域 | 建议用途 |
| --- | --- | --- |
| `esen_30m_mptrj.pt` | MPTrj | 无机晶体 PBE/PBE+U 推理和相近标注域微调 |
| `esen_30m_omat.pt` | OMat24 | 更广的无机非平衡结构 |
| `esen_30m_oam.pt` | OAM | 通用材料预训练起点 |

申请和下载入口：

- FAIR Chemistry 模型主页：https://huggingface.co/fairchem
- FAIR Chemistry 官方仓库：https://github.com/FAIR-Chem/fairchem

`Jd.pt` 随 [OneScience/eSEN](https://modelscope.cn/models/OneScience/eSEN) 模型包发布。放入本目录的 `weight/` 后，推理和微调脚本会自动使用，不需要依赖 UMA。

### 微调数据集

本示例不内置数据。oxide PBE 示例数据单独发布在 ModelScope：

```bash
modelscope download --dataset OneScience/oxides \
  --local_dir ./datasets/oxides
```

下载后的数据布局为：

```text
datasets/oxides/data/OXIDES/prepared/
├── train.db
├── val.db
└── test.db
```

数据已经转换为 ASE DB，默认配置使用 energy 和 forces 监督，不训练 stress。`prepare_oxide_dataset.py` 用于从官方 oxide JSON 重新生成该数据。

### 推理

单点能量、力和应力预测：

```bash
python single_point.py --checkpoint weight/esen_30m_mptrj.pt
```

结构弛豫：

```bash
python relax.py \
  --checkpoint weight/esen_30m_mptrj.pt \
  --fmax 0.05 --steps 100 --output relaxed.cif
```

NVT 分子动力学：

```bash
python md.py \
  --checkpoint weight/esen_30m_mptrj.pt \
  --steps 100 --temperature 300 --timestep 1.0 --output md.traj
```

通过 `--input` 可读取 CIF、POSCAR、XYZ 等 ASE 支持的结构文件。

### 微调

单卡 oxide PBE 微调：

```bash
bash demo/run.sh --config configs/finetune_1dcu.yaml
```

配置文件中的主要参数：

| YAML 字段 | 作用 |
| --- | --- |
| `checkpoint` | 初始化 checkpoint 路径 |
| `train`、`val` | ASE DB 或 ASE-LMDB 数据路径 |
| `epochs`、`batch_size`、`workers` | 训练轮数、批大小和数据加载进程数 |
| `lr` | AdamW 学习率 |
| `energy_weight`、`force_weight`、`stress_weight` | 各监督项损失权重 |
| `fit_element_references` | 是否根据训练集重新拟合能量元素参考值 |
| `launch.num_nodes`、`launch.num_gpus` | 节点数和每节点设备数 |
| `launch.mode` | `local` 直接运行；`submit` 提交 Slurm |
| `slurm.*` | Slurm 分区、时限和 CPU 资源 |

多卡和多节点使用相同入口，只需替换 YAML：

```bash
bash demo/run.sh --config configs/finetune_2dcu.yaml
bash demo/run.sh --config configs/finetune_16dcu.yaml
```

当当前节点可见设备不足，或者配置请求多个节点时，`demo/run.sh` 会自动通过 `sbatch` 提交任务。

评估微调 checkpoint：

```bash
python evaluate.py \
  --checkpoint outputs/<run>/checkpoints/esen_oxides_1dcu_finetuned.pt \
  --data datasets/oxides/data/OXIDES/prepared/test.db
```

# 文件说明

| 路径 | 作用 |
| --- | --- |
| `model/` | eSEN backbone、预测头和图接口源码 |
| `single_point.py` | 单点能量、力和应力推理 |
| `relax.py` | 周期结构弛豫 |
| `md.py` | NVT 分子动力学 |
| `finetune.py` | checkpoint 微调底层入口 |
| `evaluate.py` | 独立数据集误差评估 |
| `prepare_oxide_dataset.py` | oxide JSON 到 ASE DB 的转换脚本 |
| `demo/run.sh` | YAML 驱动的本地和 Slurm 微调入口 |
| `demo/configs/` | 单卡、多卡和多节点微调配置 |

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- eSEN 模型代码改编自 FairChem Core，遵循上游 FairChem 的 MIT License。`Jd.pt` 请按 OneScience/eSEN 模型包和上游模型协议使用。
- eSEN checkpoint 不随本仓库发布，使用时应遵循 FAIR Chemistry 模型页面给出的访问条件和许可证。
- 使用 OMat24、MPTrj、OAM 或 oxide 数据时，请分别引用实际使用的数据集和对应模型工作。
