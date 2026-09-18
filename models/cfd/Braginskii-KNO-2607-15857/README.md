<p align="center"><strong><span style="font-size: 30px;">Braginskii-KNO-2607-15857</span></strong></p>

# 模型介绍

本模型包复现 arXiv:2607.15857 中电阻率条件化 Koopman 神经算子的核心数据流：
使用目标场的两个历史帧和归一化 Spitzer 电阻率 `nu0`，联合预测未来四帧。
包内分别提供对数密度、电子温度、电势和涡量四个 fieldwise checkpoint。

由于论文的 GBS 原始轨迹未公开，Tier 1 权重在确定性小规模三维合成湍流数据上训练。
这些权重用于验证架构、训练、评估与发布流程，不代表复现论文 Table 2 的数值指标。

# 模型描述

- `model/model.py`：时间编码器、三维卷积编码/解码器和电阻率条件谱 Koopman 更新。
- `model/data.py`：可复现的三维合成湍流轨迹和两帧到四帧样本契约。
- `model/metrics.py`：MSE、R2、相对 L2、谱斜率和压力梯度长度。
- `model/toroidal.py`：论文附录 B 的最小范数环向 Fourier 重建。

# 使用说明

## 环境

```bash
pip install torch
```

## 冒烟验证

```bash
python scripts/smoke_test.py
```

## 训练

```bash
python scripts/train.py --config conf/tier1.json --output-dir outputs
```

## 评估

```bash
python scripts/evaluate.py --checkpoint-dir weight --output outputs/evaluation_metrics.json
```

## 推理

输入 `.pt` 文件需包含形状为 `[B,2,R,Z,phi]` 的 `history`、形状为 `[B]`
的 `nu0`，以及可选的 `delta_t`。

```bash
python scripts/infer.py weight/kno_theta.pt request.pt prediction.pt
```

# Tier 1 结果

四个 checkpoint 均可加载，持出 `nu0=0.1` 的 MSE、R2、相对 L2 和谱诊断均为有限值。
完整结果见 `conf/evaluation_metrics.json`，验证边界见 `conf/validation_report.md`。

# 引用

```bibtex
@article{shaa2026braginskii_kno,
  title={Surrogate modeling of drift-reduced Braginskii turbulence with resistivity-conditioned Koopman neural operators},
  author={Shaa, Ameir and Lim, Kyungtak and Chan, Long Shan and Guet, Claude},
  journal={arXiv preprint arXiv:2607.15857},
  year={2026}
}
```

论文采用 CC BY 4.0；本复现代码与模型包采用 Apache-2.0。

