from .unet import UNet
from .unetex import UNetEx


def build_model(config):
    name = config["name"] if isinstance(config, dict) else config.name
    model_map = {
        "UNet": UNet,
        "UNetEx": UNetEx,
    }
    if name not in model_map:
        raise ValueError(f"Unknown DeepCFD model: {name}")

    get = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
    return model_map[name](
        in_channels=get("in_channels"),
        out_channels=get("out_channels"),
        base_channels=get("base_channels", 16),
        num_stages=get("num_stages", 2),
        bilinear=get("bilinear", True),
        normtype=get("normtype", "bn"),
        kernel_size=get("kernel_size", 3),
    )
