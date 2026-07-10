import torch.nn as nn

from onescience.modules.decoder.unet_decoder import UNetDecoder2D
from onescience.modules.encoder.unet_encoder import UNetEncoder2D
from onescience.modules.head.unet_head import UNetHead2D


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
        num_stages: int = 2,
        bilinear: bool = True,
        normtype: str = "bn",
        kernel_size: int = 3,
    ):
        super().__init__()
        self.encoder = UNetEncoder2D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype,
            kernel_size=kernel_size,
        )
        self.decoder = UNetDecoder2D(
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype,
            kernel_size=kernel_size,
        )
        self.head = UNetHead2D(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        features = self.encoder(x)
        decoded = self.decoder(features)
        return self.head(decoded)
