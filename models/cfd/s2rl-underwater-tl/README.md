<p align="center"><strong><span style="font-size: 30px;">S2RL Underwater TL</span></strong></p>

# 模型介绍

This package implements Spectral-Spatial Residual Learning (S2RL) from “Mitigating Spectral Bias in Neural Operators for Underwater Transmission Loss Prediction” (arXiv:2608.18141). It combines a Fourier Neural Operator global propagator with a spatial U-Net residual refiner for two-dimensional range-depth transmission-loss fields.

The Tier 1 artifact uses a deterministic synthetic acoustic-like fallback because the South China Sea/FVCOM/ETOPO1/RAM dataset is not distributed with the paper. The package does not claim paper-number parity.

# 模型描述

- `model/model.py`: truncated-rFFT FNO global propagator, U-Net local refiner, and cascaded S2RL model.
- Input: `[B,3,Nz,Nr]` sound-speed, Hankel-like far-field, and bathymetry/water-mask channels.
- Output: `[B,1,Nz,Nr]` transmission loss in dB.
- Metric: water-column masked RMSE.

# 适用场景

| 场景 | 说明 |
|---|---|
| 模型训练 | `scripts/train.py` runs the two-stage Tier 1 training workflow. |
| 模型推理 | Use `S2RL` in `model/model.py` and load `weight/s2rl.pt`. |
| 评估 | `conf/metrics.json` contains the traceable masked test RMSE and training history. |

# 使用说明

## 手动安装使用

需要 Python 3.11、PyTorch 2.x；GPU/DCU 可选，CPU fallback 可运行 Tier 1。

下载模型包：

```bash
modelscope download --model OneScience/s2rl-underwater-tl
```

训练：

```bash
python scripts/train.py --config conf/config.json --out artifacts/s2rl_tier1
```

评估/冒烟：

```bash
python scripts/test_s2rl.py
```

论文数据说明：完整 South China Sea 数据、FVCOM/ETOPO1 preprocessing 和 RAM 生成流程需要用户自行提供；本包内 Tier 1 使用合成 fallback。

## 权重

- `weight/s2rl.pt`: Tier 1 trained checkpoint.
- `conf/metrics.json`: `device=cpu`, `test_rmse_db=1.3217798471450806`, `synthetic_data=true`.

# 引用与许可证

Y. Sun, S. Fang, C. Zhang, L. Cheng, J. Li, and P. Gerstoft, “Mitigating Spectral Bias in Neural Operators for Underwater Transmission Loss Prediction,” arXiv:2608.18141, 2026.

This model package is provided under Apache License 2.0. The paper-level dataset and RAM solver are external dependencies and are not bundled.

