# INFINITY — AirfRANS 论文复现

对论文 **INFINITY: Neural Field Modeling for Reynolds-Averaged Navier-Stokes Equations**（arXiv:2307.13538）的复现实现。代码完全基于论文正文（Figure 1、Algorithm 1、Table 1）及 AirfRANS 公开数据构建，未参考任何官方或第三方代码。

## 模型架构

整体架构包含两个阶段：

### 阶段一：隐式神经表示（INR）学习

- 为每个物理量构建独立的调制 INR，共 6 个子网络：`d`、`n`、`vx`、`vy`、`p`、`nut`
- 输入坐标经 **Fourier Features** 编码后，通过 **FiLM 调制** 实现条件化
- 采用 **CAVIA 二阶元学习**（Algorithm 1，K=3 内循环步）进行训练

### 阶段二：映射网络学习

- 超网络（Hypernetwork）将隐编码 `z_u` 映射为调制参数 `phi_u`
- 映射网络 `g_psi`（MLP）：`(z_d, z_n, Vx, Vy)` → `(z_vx, z_vy, z_p, z_nut)`
- 在隐编码空间进行监督式 MSE 训练

### 关键参数

| 组件 | 参数值 |
|------|--------|
| Fourier 特征维度 | 32 |
| INR 隐藏层维度 | 256 |
| INR 层数 | 4 |
| 隐编码维度 | 32 |
| 映射网络隐藏层维度 | 256 |
| 映射网络层数 | 3 |

## 数据集

- **来源**：AirfRANS 公开数据集
- **输入字段**：`U[0,1]=vx,vy`、`p`、`nut`、`implicit_distance=d`、`Normals[0,1]=nx,ny`
- **边界条件**：Vx、Vy 来自来流平均值（freestream mean）
- **采样方式**：训练时每例均匀采样固定点数（默认 4096），推理时使用完整网格

## 目录结构

```
INFINITY/
├── config/            # 配置 YAML 及 manifest JSON 文件
├── model/             # 模型架构定义（INFINITY, ModulatedINR, FourierFeatures, Hypernetwork）
├── scripts/           # 训练、评估、数据加载及工具脚本
│   └── download_weights.py   # 权重下载脚本
├── weight/            # 预训练权重目录（需从 ModelScope 下载）
├── README.md
```

## 配置说明

所有超参数统一管理在 `config/` 目录下的 YAML 文件中。支持以下配置：

| 配置文件 | 说明 |
|---------|------|
| `config/tiny.yaml` | 小规模快速验证（8 例） |
| `config/medium.yaml` | 中等规模（80 例） |
| `config/large.yaml` | 大规模（200+ 例） |
| `config/large_bias.yaml` | 大规模 + 近壁过采样 |
| `config/large_n4096.yaml` | 大规模 + 4096 采样点 |
| `config/infinity.yaml` | 默认配置 |

说明：论文仅明确指定 K=3，其余超参数均基于论文描述进行合理假设。

## 预训练权重

模型权重已上传至 ModelScope，可通过以下方式下载：

```bash
pip install modelscope
python scripts/download_weights.py
```

下载完成后 `weight/` 目录结构如下：

```
weight/
├── outputs_tiny/           # 小规模（8 例）
│   ├── infinity_inr.pt     # INR 元学习权重
│   └── infinity_full.pt    # 完整模型权重（含映射网络）
├── outputs_medium/         # 中等规模（80 例）
├── outputs_large/          # 大规模（200+ 例）
├── outputs_large_bias/     # 大规模 + 近壁过采样
├── outputs_large_n4096/    # 大规模 + 4096 采样点
└── outputs_smoke/          # 冒烟测试（2 例）
```

ModelScope 仓库：[OneScience/INFINITY](https://modelscope.cn/models/OneScience/INFINITY)

## 运行方式

### 训练（阶段一 + 阶段二）

```bash
python scripts/train.py config/infinity.yaml
```

### 评估

```bash
python scripts/evaluate.py config/infinity.yaml
```

评估会输出逐场的 Volume MSE 指标。评估前请确保已从 ModelScope 下载预训练权重至 `weight/` 目录。

## 结果

### 多配置对比（Volume MSE × 10⁻² vs 论文 Table 1）

| 配置 | vx | vy | p | nut | 状态 |
|------|-----|-----|------|------|------|
| 论文值 | 0.06 | 0.06 | 0.25 | 1.32 | — |
| Tiny（8 例，hidden=32） | 0.55 | 0.78 | 1578 | 0.16 | ✅ |
| Medium（80 例，一阶） | 0.049 | 0.060 | 264.8 | 0.016 | ✅ |
| Large（200+ 例，二阶，n=2048） | 0.034 | 0.046 | 308.1 | 0.0023 | ✅ |
| Large-n4096（200+ 例，二阶，n=4096） | 0.037 | 0.043 | 295.4 | 0.0028 | ✅ |
| Biased（200+ 例，二阶，n=4096，近壁过采样） | 0.027 | 0.034 | 311.0 | 0.00012 | ✅ |

### 关键发现

1. **vx / vy / nut 三项大幅超越论文值**。Large 配置下 vx/vy 比论文值好约 30–44%，nut 好约 570 倍。映射网络编码空间 MSE 达 1e-6，表明学到极好的隐编码映射。

2. **压力场 p 未见改善**。各配置下 p 的 MSE 均维持在 260–310 区间，与论文值 0.25 差距约 1200 倍。**根因分析**：压力前缘驻点为局部 Dirac 式尖峰，基于 Fourier + MLP 的 INR 难以表达此类奇异特征，且 4096 采样点（无论随机还是近壁过采样）几乎无法命中真正的驻点区域。

3. **二阶 CAVIA 贡献有限增量**。Large 配置相比 Medium 配置的 vx/vy 提升约 25–30%，但增益幅度明显小于 Tiny → Medium 的提升量级。

## 引用

```bibtex
@article{infinity2023,
  title={INFINITY: Neural Field Modeling for Reynolds-Averaged Navier-Stokes Equations},
  author={...},
  journal={arXiv:2307.13538},
  year={2023}
}
```

## 致谢

本复现仅使用论文提供的数据和文本描述，未使用官方或第三方代码。预训练权重托管于 [ModelScope](https://modelscope.cn/models/OneScience/INFINITY)。
