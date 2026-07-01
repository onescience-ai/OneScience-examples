from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from transformers import PretrainedConfig
except Exception:  # pragma: no cover - lets the file import in minimal environments
    class PretrainedConfig:  # type: ignore
        model_type = "custom"

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class WeatherModelConfig(PretrainedConfig):

    model_type = "mwm"

    def __init__(
        self,
        input_dim: Optional[int] = None,
        seq_len: int = 72,
        num_predict: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        encoder_type: str = "lstm",
        num_locations: int = 82,
        location_emb_dim: int = 32,
        num_weather_classes: int = 7,
        rain_pos_weight: float = 1.0,
        weather_class_weights: Optional[list[float]] = None,
        target_norms: Optional[Dict[str, Dict[str, float]]] = None,
        distill_teacher_head_dim: int = 416,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_predict = num_predict
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.num_locations = num_locations
        self.location_emb_dim = location_emb_dim
        self.num_weather_classes = num_weather_classes
        self.rain_pos_weight = rain_pos_weight
        self.weather_class_weights = weather_class_weights
        self.target_norms = target_norms or {}
        self.distill_teacher_head_dim = int(distill_teacher_head_dim)
