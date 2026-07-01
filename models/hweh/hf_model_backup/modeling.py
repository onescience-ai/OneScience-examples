from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import ModelOutput
except Exception: 
    class PreTrainedModel(nn.Module): 
        config_class = None
        base_model_prefix = ""
        main_input_name = "input_ids"

        def __init__(self, config):
            super().__init__()
            self.config = config

    class ModelOutput(dict):  # type: ignore
        pass

from .configuration import WeatherModelConfig

CONTINUOUS_TARGET_ORDER = [
    "temp",
    "humidity",
    "apparent",
    "precip",
    "sea_level_pressure",
    "surface_pressure",
    "cloud_cover",
    "wind",
    "wind_dir_sin",
    "wind_dir_cos",
]

CONTINUOUS_TARGET_SPECS = {
    "temp": {"loss_weight": 1.0, "transform": "raw"},
    "humidity": {"loss_weight": 1.0, "transform": "raw"},
    "apparent": {"loss_weight": 0.8, "transform": "raw"},
    "precip": {"loss_weight": 0.9, "transform": "log1p"},
    "sea_level_pressure": {"loss_weight": 0.6, "transform": "raw"},
    "surface_pressure": {"loss_weight": 0.4, "transform": "raw"},
    "cloud_cover": {"loss_weight": 0.4, "transform": "raw"},
    "wind": {"loss_weight": 0.6, "transform": "raw"},
    "wind_dir_sin": {"loss_weight": 0.55, "transform": "raw"},
    "wind_dir_cos": {"loss_weight": 0.55, "transform": "raw"},
}


@dataclass
class WeatherModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[Tuple[torch.Tensor, ...]] = None
    head_repr: Optional[torch.Tensor] = None
    norm_preds: Optional[Dict[str, torch.Tensor]] = None
    raw_preds: Optional[Dict[str, torch.Tensor]] = None
    distill_head_repr: Optional[torch.Tensor] = None


class WeatherForcastModel(PreTrainedModel):

    config_class = WeatherModelConfig
    base_model_prefix = "weather_sequence"
    main_input_name = "X"

    # Newer Transformers versions may create auto_map entries from these registrations.
    _tied_weights_keys: list[str] = []

    def __init__(self, config: WeatherModelConfig):
        super().__init__(config)

        self.encoder_type = str(getattr(config, "encoder_type", "lstm")).lower()
        self.hidden_dim = int(config.hidden_dim)
        self.seq_len = int(config.seq_len)
        self.num_predict = int(config.num_predict)
        self.num_weather_classes = int(config.num_weather_classes)

        if config.input_dim is None:
            raise ValueError("WeatherModelConfig.input_dim must be set")

        self.location_embedding = nn.Embedding(max(1, int(config.num_locations)), int(config.location_emb_dim))

        if config.weather_class_weights is not None:
            self.register_buffer(
                "weather_class_weights",
                torch.tensor(config.weather_class_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.weather_class_weights = None

        self.register_buffer(
            "rain_pos_weight",
            torch.tensor(float(config.rain_pos_weight), dtype=torch.float32),
            persistent=False,
        )

        self.target_norm_meta: Dict[str, Dict[str, Any]] = {}
        for name in CONTINUOUS_TARGET_ORDER:
            spec = dict(config.target_norms.get(name, {}))
            mean = float(spec.get("mean", 0.0))
            std = max(float(spec.get("std", 1.0)), 1e-6)
            transform = str(spec.get("transform", CONTINUOUS_TARGET_SPECS[name]["transform"]))
            self.register_buffer(f"{name}_mean", torch.tensor(mean, dtype=torch.float32), persistent=False)
            self.register_buffer(f"{name}_std", torch.tensor(std, dtype=torch.float32), persistent=False)
            self.target_norm_meta[name] = {"transform": transform}

        if self.encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=int(config.input_dim),
                hidden_size=self.hidden_dim,
                num_layers=int(config.num_layers),
                batch_first=True,
                dropout=float(config.dropout) if int(config.num_layers) > 1 else 0.0,
                bidirectional=False,
            )
        elif self.encoder_type == "transformer":
            self.input_proj = nn.Linear(int(config.input_dim), self.hidden_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, int(config.seq_len), self.hidden_dim) * 0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=4,
                dropout=float(config.dropout),
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(config.num_layers))
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        self.head_dim = self.hidden_dim + int(config.location_emb_dim)
        self.head_norm = nn.LayerNorm(self.head_dim)
        self.head_dropout = nn.Dropout(float(config.dropout))

        self.reg_heads = nn.ModuleDict({name: nn.Linear(self.head_dim, self.num_predict) for name in CONTINUOUS_TARGET_ORDER})
        self.fc_rain = nn.Linear(self.head_dim, self.num_predict)
        self.fc_weather = nn.Linear(self.head_dim, self.num_predict * self.num_weather_classes)

        teacher_head_dim = int(getattr(config, "distill_teacher_head_dim", 0))
        if teacher_head_dim > 0 and teacher_head_dim != self.head_dim:
            self.distill_proj = nn.Linear(self.head_dim, teacher_head_dim, bias=False)
        else:
            self.distill_proj = None

        self.post_init()

    @staticmethod
    def _masked_mean(x: torch.Tensor) -> torch.Tensor:
        mask = (x.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _target_mean_std(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return getattr(self, f"{name}_mean"), getattr(self, f"{name}_std")

    def _encode_target(self, name: str, target: torch.Tensor) -> torch.Tensor:
        transform = self.target_norm_meta[name]["transform"]
        target = target.to(dtype=torch.float32)
        if transform == "log1p":
            target = torch.log1p(torch.clamp(target, min=0.0))
        mean, std = self._target_mean_std(name)
        return (target - mean.to(target.device)) / std.to(target.device)

    def _decode_prediction(self, name: str, pred_norm: torch.Tensor) -> torch.Tensor:
        transform = self.target_norm_meta[name]["transform"]
        mean, std = self._target_mean_std(name)
        raw = pred_norm * std.to(pred_norm.device) + mean.to(pred_norm.device)
        if transform == "log1p":
            raw = torch.expm1(raw).clamp(min=0.0)
        return raw

    def forward(
        self,
        X: torch.Tensor,
        location_id: Optional[torch.Tensor] = None,
        temp_target: Optional[torch.Tensor] = None,
        humidity_target: Optional[torch.Tensor] = None,
        apparent_target: Optional[torch.Tensor] = None,
        precip_target: Optional[torch.Tensor] = None,
        sea_level_pressure_target: Optional[torch.Tensor] = None,
        surface_pressure_target: Optional[torch.Tensor] = None,
        cloud_cover_target: Optional[torch.Tensor] = None,
        wind_target: Optional[torch.Tensor] = None,
        wind_dir_sin_target: Optional[torch.Tensor] = None,
        wind_dir_cos_target: Optional[torch.Tensor] = None,
        rain_target: Optional[torch.Tensor] = None,
        weather_target: Optional[torch.Tensor] = None,
        return_repr: bool = False,
        **kwargs: Any,
    ) -> WeatherModelOutput:
        if location_id is None:
            location_id = torch.zeros(X.size(0), dtype=torch.long, device=X.device)

        if self.encoder_type == "lstm":
            _, (h, _) = self.encoder(X)
            seq_repr = h[-1]
        else:
            z = self.input_proj(X) + self.pos_encoding[:, : X.size(1), :]
            out = self.encoder(z)
            seq_repr = self._masked_mean(out)

        loc_emb = self.location_embedding(location_id)
        head_repr = self.head_norm(torch.cat([seq_repr, loc_emb], dim=1))
        h = self.head_dropout(head_repr)

        raw_preds: Dict[str, torch.Tensor] = {}
        norm_preds: Dict[str, torch.Tensor] = {}
        for name in CONTINUOUS_TARGET_ORDER:
            norm_pred = self.reg_heads[name](h)
            norm_preds[name] = norm_pred
            raw_preds[name] = self._decode_prediction(name, norm_pred)

        rain_logit = self.fc_rain(h)
        weather_logits = self.fc_weather(h).view(-1, self.num_predict, self.num_weather_classes)

        loss = None
        if temp_target is not None:
            targets = {
                "temp": temp_target,
                "humidity": humidity_target,
                "apparent": apparent_target,
                "precip": precip_target,
                "sea_level_pressure": sea_level_pressure_target,
                "surface_pressure": surface_pressure_target,
                "cloud_cover": cloud_cover_target,
                "wind": wind_target,
                "wind_dir_sin": wind_dir_sin_target,
                "wind_dir_cos": wind_dir_cos_target,
            }

            loss_terms = []
            for name, target in targets.items():
                if target is None:
                    continue
                target_norm = self._encode_target(name, target.to(h.device))
                pred_norm = norm_preds[name].to(target_norm.dtype)
                loss_terms.append(
                    F.smooth_l1_loss(pred_norm, target_norm) * float(CONTINUOUS_TARGET_SPECS[name]["loss_weight"])
                )

            if rain_target is not None:
                rain_target = rain_target.to(rain_logit.dtype)
                rain_loss = F.binary_cross_entropy_with_logits(
                    rain_logit,
                    rain_target,
                    pos_weight=self.rain_pos_weight.to(rain_logit.device),
                )
                loss_terms.append(0.7 * rain_loss)

            if weather_target is not None:
                weather_loss = F.cross_entropy(
                    weather_logits.reshape(-1, self.num_weather_classes),
                    weather_target.long().reshape(-1),
                    weight=self.weather_class_weights,
                    label_smoothing=0.0,
                )
                loss_terms.append(0.9 * weather_loss)

            loss = sum(loss_terms) if loss_terms else None

        logits = (
            raw_preds["temp"],
            raw_preds["humidity"],
            raw_preds["apparent"],
            raw_preds["precip"],
            raw_preds["sea_level_pressure"],
            raw_preds["surface_pressure"],
            raw_preds["cloud_cover"],
            raw_preds["wind"],
            raw_preds["wind_dir_sin"],
            raw_preds["wind_dir_cos"],
            rain_logit,
            weather_logits,
        )

        output = WeatherModelOutput(
            loss=loss,
            logits=logits,
            head_repr=head_repr if return_repr else None,
            norm_preds=norm_preds if return_repr else None,
            raw_preds=raw_preds if return_repr else None,
            distill_head_repr=(self.distill_proj(head_repr) if self.distill_proj is not None else head_repr) if return_repr else None,
        )
        return output


# Make the repo usable with AutoConfig/AutoModel when loaded from the Hub.
try:  # pragma: no cover
    WeatherModelConfig.register_for_auto_class()
except Exception:
    pass

try:  # pragma: no cover
    WeatherForcastModel.register_for_auto_class("AutoModel")
except Exception:
    pass
