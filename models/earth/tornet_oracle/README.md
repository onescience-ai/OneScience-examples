---
library_name: pytorch
license: mit
datasets:
- TorNet
tags:
- weather
- radar
- tornado
- tornado_prediction
- NEXRAD
- MRMS
- HRRR
- lightning
metrics:
- auprc
- f1
- accuracy
- brier
- ece
pipeline_tag: image-classification
language:
- en
---

# Wonder-Griffin/tornado-super-predictor

**TornadoSuperPredictor** from Storm-Oracle, trained on **TorNet (Zenodo)** patches.  
Outputs a tornado probability per patch (optionally with atmospheric features).

## Summary

- **Data**: TorNet (official split); optional recent holdout recommended.  
- **Architecture**: CNN feature extractor + heads (probability, EF logits, location, timing, uncertainty).  
- **Temporal**: 3 volume(s) stacked as channels.  
- **Normalization**: zscore.  
- **Loss**: bce (pos_weight=2.0).  
- **Calibration**: Platt (A,B)=n/a,n/a; Temperature T=n/a.

## Intended Use

- Research on tornado nowcasting from radar patches;  
- Evaluation under class imbalance with PR metrics;  
- **Not** an operational warning system without further validation & human oversight.

## Dataset

- **Train examples**: 6  
- **Eval examples**: 4  
- **Class balance**: positives=n/a, negatives=n/a, pos_weight≈2.0

## Evaluation (threshold = 0.5)

Confusion matrix (rows = truth, cols = prediction):

|        | Pred 0 | Pred 1 |
|-------:|-------:|-------:|
| True 0 | 0 | 2 |
| True 1 | 0 | 2 |

Metrics:

- **AUPRC**: n/a  
- **Accuracy**: n/a  
- **(Optional)**: attach PR curve & reliability diagrams

## Training

- Optimizer: AdamW (lr=1e-4, wd=1e-4 by default)  
- Batch size: n/a  
- Epochs: n/a  
- Precision: 16-mixed  
- Augmentations: flips/rotations/intensity jitter + optional crops  
- Hardware: 1× GPU (FP16 mixed)

## Quickstart

```python
import torch
from transformers import AutoModel

repo = "Wonder-Griffin/TorNet-Oracle"
model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()

# Example dummy batch
B, T, H, W = 2, 1, 256, 256  # T time steps -> in_channels = 3*T (reflectivity, velocity, spectrum width?)
radar_x = torch.randn(B, 3*T, H, W)

# Atmospheric dictionary (use only what you have; shapes must be (B, dim))
atmo = {
    "cape":        torch.randn(B, 1),
    "wind_shear":  torch.randn(B, 4),  # 0–1, 0–3, 0–6, deep
    "helicity":    torch.randn(B, 2),  # 0–1, 0–3
    "temperature": torch.randn(B, 3),  # sfc, 850, 500
    "dewpoint":    torch.randn(B, 2),  # sfc, 850
    "pressure":    torch.randn(B, 1),
}

out = model(radar_x=radar_x, atmo=atmo)
print(out.tornado_probability.shape)  # (B,)
print(out.ef_scale_probs.shape)       # (B, 6)
print(out.location_offset.shape)      # (B, 2)
print(out.timing_predictions.shape)   # (B, 3)
---

# 3) Notes to avoid common gotchas

- **Export the class names**: Make sure `StormOracleModel` and `StormOracleConfig` are importable at the repo root via `__init__.py`. Hugging Face uses that when `trust_remote_code=True`.
- **Architectures**: The `"architectures"` array in `config.json` **must** include `"StormOracleModel"`.
- **Weights**: You already have `pytorch_model.bin`/**or** `model.safetensors`. Either is fine. Keep the filenames standard.
- **Forward signature**: With remote code, it’s okay that `forward` takes `radar_x` and `atmo`. Users pass them as keyword args as shown.
- **Version pins**: If you rely on features from newer `transformers`, keep the `transformers_version` in `config.json` current.

---

# 4) Optional niceties

- **`hubconf.py`** (for `torch.hub` users):
  ```python
  from .tornado_predictor import TornadoSuperPredictor

  def storm_oracle(in_channels=3, pretrained=False, hf_repo=None, map_location="cpu"):
      model = TornadoSuperPredictor(in_channels=in_channels)
      if pretrained and hf_repo is not None:
          from huggingface_hub import hf_hub_download
          path = hf_hub_download(hf_repo, filename="pytorch_model.bin")
          import torch
          state = torch.load(path, map_location=map_location)
          model.load_state_dict(state, strict=True)
      return model