# 模型介绍

MARIO（Modulated Aerodynamic Resolution-Independent Operator）是 Giovannni Catalani 等人在论文 *MARIO: A Modulated Neural Field Framework for Resolution-Independent Surrogate Modeling of Aerodynamics* 中提出的气动代理模型框架。模型使用条件 Neural Field 表示二维翼型流场，通过几何 SDF latent 和来流条件调制解码器，在训练时随机采样网格点，在推理时可对完整非结构网格进行查询。


## 模型架构

本复现工程包含两个主要阶段：

1. **Geometry Encoder**
   - 输入：二维坐标 `(x, y)` 和 SDF。
   - 结构：带 Fourier features 的 modulated neural field。
   - 输出：每个翼型 case 的低维几何 latent。
   - 训练目标：SDF reconstruction MSE。

2. **MARIO Decoder**
   - 输入：`x, y, sdf, nx, ny, boundary-layer mask`。
   - 条件：`geometry latent + freestream Vx,Vy`。
   - 输出：`ux, uy, p, nut`。
   - 训练目标：标准化流场 MSE，并支持 surface pressure、边界层区域和通道权重。

# 仓库说明
本工程基于 OneScience 技能整理并复现 MARIO 在 AirfRANS scarce setting 上的二维翼型实验，适用于小样本气动代理建模、二维翼型稳态 RANS 流场预测、表面压力和体场联合评估等场景。


当前支持能力：

- 训练 MARIO geometry encoder 和 decoder。
- 推理并在完整 AirfRANS mesh 上评估 `ux/uy/p/nut/surface_p`。
- 读取 AirfRANS VTK XML 数据并进行 case cache、统计量标准化和随机点采样。
- checkpoint resume、validation-best checkpoint 选择、早停和指标 JSON 输出。
- 提供 DCU 环境依赖文件和 运行脚本。

当前不支持能力：

- 不自动下载 AirfRANS 数据集，需用户提前准备 `manifest.json` 和对应 `.vtu/.vtp` 文件。
- 不内置图形化流场可视化脚本；当前评估输出为 JSON 指标。
- 不覆盖论文中的 NASA CRM 实验。
- 最终归档未重新计算论文 Table 3 的 `CD/CL` force metrics。

# 适用场景

| 场景 | 说明 |
| ---------- | --------------------------------- |
| 二维翼型气动代理建模 | 面向 AirfRANS 风格二维翼型稳态 RANS 数据，预测速度、压力和湍流黏度 |
| 小样本流场预测 | 复现论文 scarce setting，使用约 200 个训练 case 训练代理模型 |
| 非结构网格场查询 | 训练时随机采样点，推理时可在完整 mesh 上分块查询 |
| 表面压力评估 | 额外统计 surface pressure MSE，用于分析近壁区域和气动力相关误差 |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | 工程使用说明文档 | 中文为主 |
| `requirements_dcu.txt` | DCU 运行环境依赖 | 包含 DAS/DTK 适配 PyTorch wheel 和常用科学计算依赖 |
| `src/` | MARIO AirfRANS 源码 | 当前 Python 包入口，使用 `python -m src.train/evaluate` |
| `src/data.py` | AirfRANS 数据读取和采样 | 读取 VTK XML、构造 SDF/normals/boundary-layer mask |
| `src/model.py` | 模型模块 | Fourier features、hypernetwork、GeometryEncoder、MarioDecoder |
| `src/train.py` | 训练脚本 | 支持 geometry/decoder/all、resume、validation best、早停 |
| `src/evaluate.py` | 推理和评估脚本 | 输出 full_test 指标 JSON，可选 force metrics |
| `src/vtk_xml.py` | VTK XML 轻量读取器 | 解析 `.vtu/.vtp` 数据 |
| `configs/airfrans_mario_e02r1n06_paper_aligned.json` | 本次 500 epoch 复现配置 | 指向当前 `mario/outputs` |
| `configs/airfrans_mario_strict.json` | 论文 strict 参考配置 | 1000 epoch 参考设置 |
| `configs/airfrans_mario_smoke_verify.json` | 快速验证配置 | 仅用于流程连通性验证，不保存 smoke 结果 |
| `scripts/run_e02r1n06_paper_aligned.sh` | 训练和评估脚本 | 运行 decoder/evaluate |
| `scripts/guarded_continue.py` | 早期调试脚本 | full_test 守门策略，仅保留作实验记录 |
| `outputs/mario_e02r1n06_paper_aligned/` | 本次正式输出 | 包含 `decoder_best.pt`、`geometry_last.pt`、stats、latents 和 full_test 指标 |

用户可通过魔搭社区下载预训练好的模型权重进行推理微调：
```
modelscope download --model OneScience/MARIO  outputs --local_dir ./outputs
```

# 使用说明

## 1. OneCode 使用

可通过 OneCode 在线环境体验智能化一键式 AI4S 编程：

[点击体验智能化一键式 AI4S 编程](https://web-2069360198568017922-iaaj.ksai.scnet.cn:58043/home)

## 2. 手动安装使用

**硬件要求**

- 推荐使用 GPU 或 DCU 运行。
- DCU 用户需要预先安装 DTK，建议使用 DTK 25.04.2 以上版本或与当前集群匹配的 OneScience 推荐版本。


### 安装运行环境

```bash
# 激活DTK及CONDA
conda create -n onescience311 python=3.11 -y
conda activate onescience311
pip install onescience[cfd] -i http://mirrors.onescience.ai:3141/pypi/simple/  --trusted-host mirrors.onescience.ai
```

## 3. 快速开始

### 下载模型包

```bash
modelscope download --model OneScience/MARIO --local_dir ./mario
cd mario
```


### 数据准备

OneScience 社区提供可供训练的 airfrans 数据，用户可通过下述命令下载：

```bash
modelscope download --dataset OneScience/airfrans --local_dir ./data
```

### 训练

快速流程验证：

```bash
python -m src.train --config configs/airfrans_mario_smoke_verify.json --stage all
```

复现实验继续训练/评估：

```bash
bash scripts/run_e02r1n06_paper_aligned.sh
```


### 推理和可视化

当前推理入口会在完整 mesh 上计算指标并写出 JSON：

```bash
python -m src.evaluate --config configs/airfrans_mario_e02r1n06_paper_aligned.json
```

输出文件：

```text
outputs/mario_e02r1n06_paper_aligned/eval/full_test_metrics.json
```

当前仓库不内置流场图片可视化脚本；如需可视化，可基于 VTK/ParaView 或自定义脚本读取原始网格和模型预测结果扩展。

## OneSkills复现说明

### 与原论文对比

| 项目 | 原论文 MARIO | 当前 mario 复现 |
| --- | --- | --- |
| 数据 split | AirfRANS scarce 200 train-val / 200 test | scarce_train=200, full_test=200 |
| 点采样 | 每 case 随机 16000 点 | 每 case 16000 点 |
| 模型主结构 | 条件 neural field + geometry latent + modulation | 同方向复现，简化/工程化实现 |
| Fourier / BL mask | sigma=1, tau=0.02 | sigma=1.0, tau=0.02 |
| decoder 训练轮数 | 约 1000 epochs | 当前归档到 500 epochs |
| 优化器/学习率 | 论文记录 Adam lr=1e-3 | adamw lr=2.5e-05，用于 continuation 稳定训练 |
| 验证策略 | 论文未给完整 checkpoint selection 细节 | 20-case monitor，holdout=false，best checkpoint restore |
| 硬件 | NVIDIA A100 | e02r1n06 / BW |
| Table 2 指标 | MARIO: ux=0.152, uy=0.113, p=0.240, nu_t=0.096, p_s=0.270 | full_test 仍为论文的 19.8x-108.6x |
| Table 3 force | CD MRE=0.794%, CL MRE=0.115%, rho_D=0.102, rho_L=0.997 | 当前归档未完成同口径 force metric |

### 复现的实验设置

| 项目 | 设置 |
| :--- | :--- |
| 数据集 | AirfRANS 风格二维翼型数据 |
| 训练 split | `scarce_train`，200 cases |
| 测试 split | `full_test`，200 cases |
| Geometry encoder | latent dim 8，hidden 256，5 层，64 Fourier features |
| Decoder | hidden 256，5 层，3-layer hypernetwork |
| 输入 | `x,y,sdf,nx,ny,boundary-layer mask` + `geometry latent,Vx,Vy` |
| 输出 | `ux,uy,p,nut` |
| 训练点数 | 每 case 随机采样 16000 点 |
| Geometry epoch | 100 |
| Decoder epoch | 500 |
| 环境 | `source ~/env.sh` |

### 训练收敛趋势

![decoder training loss](logs/figures/mario_decoder_training_loss.svg)

训练曲线说明：
- `train_loss` 总体下降明显，`surface_p_loss` 同步下降，是后期收敛的主要贡献之一。
- `field_loss` 绝对值较小，后期改善幅度有限；说明模型在全场标准化 MSE 上继续压低的空间比 surface pressure 更小。
- `epoch=500` 同时是训练 loss 和 validation score 的最好点，说明在当前 validation monitor 上尚未出现回升。

### 复现的结果对比

当前工程已经完成从 AirfRANS 数据读取、几何编码、decoder 训练、checkpoint 保存到 full mesh 推理评估的完整流程，可作为 OneScience / ModelScope 场景下的可运行复现案例。

需要说明的是，本次归档的 validation 更偏向训练过程监控，尚不能等同于严格独立泛化评估。与原论文报告的最佳结果相比，当前复现仍存在较大差距，主要原因可能来自复现实现细节、数据读取与采样协议、训练策略、验证集划分以及气动力后处理口径尚未完全对齐。

因此，当前版本更适合作为“论文方法工程化落地与流程连通性验证”的案例：它展示了如何将 MARIO 思路整理成标准工程，并提供训练、推理、评估入口；若要作为严格论文精度复现，还需要继续补齐评估协议和训练细节。

# 引用与许可证

- MARIO 原始论文：[MARIO: A Modulated Neural Field Framework for Resolution-Independent Surrogate Modeling of Aerodynamics](https://arxiv.org/abs/2505.14704)
- AirfRANS 数据集：请遵循原数据集发布方许可和使用条款。
- 本仓库保留来源说明，并面向 OneScience ModelScope 自动运行场景进行整理；公开分发前请根据上游项目确认许可证要求。
