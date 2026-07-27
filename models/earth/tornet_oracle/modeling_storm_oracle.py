import torch
import torch.nn as nn
from transformers import PreTrainedModel
from configuration_storm_oracle import StormOracleConfig

# ---- import your actual model code ----
# If your code lives in tornado_predictor.py (as pasted), import from there:
from tornado_predictor import TornadoSuperPredictor  # adjust if filename differs

class StormOracleModel(PreTrainedModel):
    config_class = StormOracleConfig

    def __init__(self, config: StormOracleConfig):
        super().__init__(config)
        self.model = TornadoSuperPredictor(in_channels=config.in_channels)
        self.post_init()  # HF bookkeeping

    def forward(self, radar_x: torch.Tensor, atmo: dict):
        """
        radar_x: (B, C, H, W)
        atmo: dict of tensors (cape, wind_shear, helicity, temperature, dewpoint, pressure)
        returns TornadoPredictionBatch (your dataclass)
        """
        return self.model(radar_x, atmo)
