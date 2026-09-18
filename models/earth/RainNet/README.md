<p align="center">
  <strong>
    <span style="font-size: 30px;">RainNet</span>
  </strong>
</p>

# 模型介绍

论文：RainNet v1.0: a convolutional neural network for radar-based precipitation nowcasting  
https://doi.org/10.5194/gmd-13-2631-2020

RainNet 用于雷达降水短临预报，输入过去四个连续的 5 分钟雷达降水场，预测未来 5 分钟的降水场，并可通过递归推理扩展到约 60 分钟。

# 模型描述

RainNet 由原论文作者团队提出，使用德国气象局 DWD 的 RY 雷达降水产品训练。模型用于连续降水强度的雷达临近预报。

当前实现将 4 个连续历史帧作为输入，并以紧随其后的时间步 `i+4` 作为 Target。降水值进入模型前按 `x → log(x + 0.01)` 变换；原始 `900×900` 雷达场使用 reflect/mirror padding 扩展为 `928×928`，模型输出后再裁剪回 `900×900`。RainNet 约含 31.38M 参数，Decoder 使用 nearest-neighbor upsampling。

# 适用场景

| 场景 | 说明 |
| :---: | :--- |
| 连续降水回归训练 | 使用连续雷达降水场训练 RainNet。 |
| 雷达降水临近预报 | 使用历史连续雷达场预测未来降水。 |
| 本地快速验证 | 使用 Fake Data 检查数据读取、模型训练、推理、评估和可视化。 |
| ModelScope / OneCode 运行 | 作为独立模型包运行。 |
| 多卡训练 | 通过 `torchrun` 启动多进程训练。 |

# 使用说明

以下命令均在模型包根目录执行。默认配置使用完整 `900×900` 空间网格、1 个 epoch 和每阶段最多 1 个 batch 完成工程冒烟验证。

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

需要 Python 3.11，以及 PyTorch、NumPy、h5py、PyYAML 和 Matplotlib。安装环境后依次执行 Fake Data 生成、训练、推理与评估命令。

### 硬件要求

模型约含 31.4M 参数。完整 `928×928` 内部张量的训练内存开销较高，建议使用具有充足显存的 CUDA/HIP 兼容 GPU 或 DCU；也支持 CPU，但运行时间会明显增加。

### 下载模型包

```bash
modelscope download --model OneScience/RainNet --local_dir ./RainNet
cd RainNet
```

### 安装运行环境

#### DCU 环境

```bash
# 请首先激活 DTK 及 CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311

pip install onescience[earth-dcu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

#### GPU 环境

```bash
# 请首先激活 CONDA
conda create -n onescience311 python=3.11 -y \
  libstdcxx-ng=12 \
  libgcc-ng=12 \
  gcc_linux-64=12 \
  gxx_linux-64=12

conda activate onescience311

pip install onescience[earth-gpu] \
  -i http://mirrors.onescience.ai:3141/pypi/simple/ \
  --trusted-host mirrors.onescience.ai
```

### 训练数据介绍

#### 真实数据

真实数据为 RYDL，官方地址为 https://doi.org/10.5281/zenodo.3629951。数据采用 HDF5 格式，原始单帧为 `900×900`，空间分辨率为 1 km，时间分辨率为 5 min。HDF5 顶层的每个时间 key 对应一个二维降水场。本仓库不包含完整真实数据，也不会自动下载该数据。

#### Fake Data

```bash
python scripts/fake_data.py
```

该命令生成 `data/rainnet_fake.hdf5`。Fake Data 保持真实单帧 `900×900`，模拟真实 RYDL HDF5 key-value 组织，保持 5 分钟连续时间序列，只缩减时间帧数量。Fake Data 的实验结果只用于验证工程链路，不代表真实降水预报精度，也不代表论文精度复现。

### 训练

当前复现使用 Log-Cosh Loss 和 Adam Optimizer，默认 learning rate 为 `1e-4`。

#### 单卡训练

```bash
python scripts/fake_data.py
python scripts/train.py
```

#### 多卡训练

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --rdzv_id=1000 \
  --rdzv_backend=c10d \
  --max_restarts=0 \
  --master_addr="localhost" \
  --master_port=29500 \
  scripts/train.py
```

### 训练权重

本仓库计划在 weight/ 目录提供基于 DWD RY/RYDL 雷达降水数据训练的权重，权重文件即将上传，预计将于近期完成。

### 推理

```bash
python scripts/inference.py
```

推理结果保存到 `result/output/`，包含单步结果和 12 步递归结果。推理必须加载训练生成的 checkpoint。

### 评估和可视化

```bash
python scripts/result.py
```

脚本从真实推理输出计算 MAE、CSI、FSS 和 Persistence Baseline，并生成预测可视化、损失曲线和指标图。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- 当前 OneScience RainNet PyTorch 模型包在模型卡 YAML 元数据中标记为 Apache License 2.0。
- RainNet 原作者官方源码采用 MIT License。
- RainNet 论文由 Copernicus Publications 出版，论文正式页面标明采用 Creative Commons Attribution 4.0 License（CC BY 4.0）。
- 使用与再发布时必须保留原论文和原作者归属。

```bibtex
@article{ayzel2020rainnet,
  title={RainNet v1.0: a convolutional neural network for radar-based precipitation nowcasting},
  author={Ayzel, Georgy and Scheffer, Tobias and Heistermann, Maik},
  journal={Geoscientific Model Development},
  volume={13},
  pages={2631--2644},
  year={2020},
  doi={10.5194/gmd-13-2631-2020}
}
```
