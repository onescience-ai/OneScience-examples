# DRIFT-2607-14394

这是 arXiv:2607.14394 DRIFT 的 Tier 1 可运行复现包，包含 DTST 直接保留模式 DFT、DRIFT 频谱块、训练、推理和评估脚本。论文 PDEBench 原始数据、多节点 GPU 环境和性能 benchmark 未包含在此包中；指标仅对应确定性合成周期 PDE fallback。

## 使用

```bash
python scripts/smoke_test.py
python scripts/train.py
python scripts/evaluate.py
python scripts/infer.py
```

论文来源：<https://arxiv.org/abs/2607.14394>

