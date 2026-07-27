from transformers import PretrainedConfig

class StormOracleConfig(PretrainedConfig):
    model_type = "storm_oracle"

    def __init__(self,
                 in_channels: int = 3,
                 image_size: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.image_size = image_size
