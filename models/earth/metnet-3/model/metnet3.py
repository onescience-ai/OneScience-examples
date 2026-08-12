from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .metnet3_blocks import ConvBlock, LongRangeMaxViT
from .metnet3_heads import ClassificationHead, RegressionHead
from .metnet3_schema import InputSchema, validate_batch


@dataclass
class MetNet3Config(InputSchema):
    hidden: int = 32  # ADAPTATION: compact width versus the paper's production model.
    maxvit_blocks: int = 1  # ADAPTATION
    condition_dim: int = 16


class MetNet3(nn.Module):
    def __init__(self, config: MetNet3Config | None = None):
        super().__init__()
        self.config = config or MetNet3Config()
        s = self.config
        input_channels = s.high_channels + s.low_channels + s.omo_channels + s.hrrr_channels + s.goes_channels + s.total_static_channels
        self.condition = nn.Sequential(nn.Linear(2, s.condition_dim), nn.GELU(), nn.Linear(s.condition_dim, s.condition_dim))
        self.enc1 = ConvBlock(input_channels, s.hidden, s.condition_dim)
        self.down = nn.Conv2d(s.hidden, s.hidden * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(s.hidden * 2, s.hidden * 2, s.condition_dim)
        self.long_range = LongRangeMaxViT(s.hidden * 2, blocks=s.maxvit_blocks)
        self.up = nn.ConvTranspose2d(s.hidden * 2, s.hidden, 2, stride=2)
        self.dec = ConvBlock(s.hidden * 2, s.hidden, s.condition_dim)
        self.precipitation = ClassificationHead(s.hidden, 1, s.precipitation_bins)
        self.ground = ClassificationHead(s.hidden, s.ground_targets, s.ground_bins)
        self.hrrr = RegressionHead(s.hidden, s.hrrr_channels)

    @staticmethod
    def _last_frame(x: torch.Tensor) -> torch.Tensor:
        return x[:, -1]

    def forward(self, batch: Mapping[str, torch.Tensor]):
        s = self.config
        validate_batch(batch, s)
        high = self._last_frame(batch["mrms_high"])
        low = self._last_frame(batch["mrms_low"])
        if low.shape[-2:] != high.shape[-2:]:
            low = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        omo = self._last_frame(batch["omo"]) * batch["omo_input_mask"].float()
        hrrr = self._last_frame(batch["hrrr"])
        goes = self._last_frame(batch["goes"])
        static = torch.cat([batch["elevation"], batch["coordinates"], batch["topography_embedding"]], dim=1)
        x = torch.cat([high, low, omo, hrrr, goes, static], dim=1)
        condition = self.condition(torch.cat([batch["current_time"], batch["lead_time"]], dim=1).float())
        skip = self.enc1(x, condition)
        deep = self.enc2(self.down(skip), condition)
        deep = self.long_range(deep)
        decoded = self.up(deep)
        decoded = self.dec(torch.cat([decoded, skip], dim=1), condition)
        return {"precipitation_logits": self.precipitation(decoded), "ground_logits": self.ground(decoded), "hrrr_regression": self.hrrr(decoded)}
