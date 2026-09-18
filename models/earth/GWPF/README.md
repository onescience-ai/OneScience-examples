<p align="center"><strong><span style="font-size: 30px;">GWPF</span></strong></p>

# 模型介绍

GWPF（Fourier Geometric Wind Power Forecasting with Numerical Weather Prediction）复现模型，
来自 KDD '26 论文《Fourier Geometric Wind Power Forecasting with Numerical Weather Prediction》
(arXiv:2607.17095)。多模态风电功率预测框架：融合历史逐点 SCADA 数据与网格化 NWP（CERRA）数值天气预报，
显式分解标量/向量特征，几何编码器提取旋转不变风向量特征，FNO 频域融合建模长程时空依赖。

# 模型描述

- model/ 下包含 GWPF 模型实现（几何编码器、频域融合、MLP 解码器）

# 适用场景

| 场景 | 说明 |
|------|------|
| 短时风电功率预测 | 基于 SCADA 历史 + NWP 预报预测未来 1-6 小时涡轮功率 |
| 模型推理 | 加载权重进行功率预测 |

# 使用说明

## OneCode 使用

通过 OneScience 平台加载该模型包。

## 手动安装使用

### 硬件要求

GPU / DCU / CPU（PyTorch 2.x）

### 下载模型包

```bash
modelscope download --model OneScience/GWPF
```

### 安装运行环境

```bash
conda create -n gwpf python=3.11
pip install torch numpy pyyaml
```

### 训练数据介绍

Tier 1 使用合成/仿真 SCADA+CERRA 风格多风场数据（Kelmarsh/Penmanshiel/Hill of Towie 结构），
真实三风电场公开数据可参考论文引用来源。

### 训练

```bash
python scripts/train.py --config conf/config.yaml --epochs 60
```

### 训练权重

- gwpf_Hill of Towie_N21_K21_tier1.pt
- gwpf_Kelmarsh_N6_K16_tier1.pt
- gwpf_Penmanshiel_N14_K20_tier1.pt

### 推理

```bash
python scripts/infer.py --ckpt weight/gwpf_Kelmarsh_N6_K16_tier1.pt --farm Kelmarsh
```

### 评估和可视化

```bash
python scripts/eval.py --ckpt weight/gwpf_Kelmarsh_N6_K16_tier1.pt --farm Kelmarsh
```

# OneScience 官方信息

| 项目 | 链接 |
|------|------|
| OneScience | https://onescience.ai |

# 引用与许可证

Piao et al. "Fourier Geometric Wind Power Forecasting with Numerical Weather Prediction."
KDD '26. arXiv:2607.17095. Apache License 2.0.

