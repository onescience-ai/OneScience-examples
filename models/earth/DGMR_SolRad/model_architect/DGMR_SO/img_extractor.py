import torch.nn as nn

from ..components.common import DBlock


class ImageExtractor(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            apply_down_flag,
            down_step=4):
        """
        in_c -> 1
        x) base_c is 1/96 of out_channels
        base_c is set to 4
        """
        super().__init__()
        self.down_step = down_step

        self.base_c = out_channels // 96
        if self.base_c < 4:
            self.base_c = 4
        cc = self.base_c

        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=2)

        chs = [in_channels * 4, cc * 3, cc * 6, cc * 24, out_channels]
        self.DList = nn.ModuleList()
        for i in range(down_step):
            self.DList.append(
                DBlock(
                    in_channel=chs[i],
                    out_channel=chs[i + 1],
                    conv_type='2d',
                    apply_down=apply_down_flag[i]
                ),
            )

    def forward(self, x):
        """
        x
        """
        y = self.space_to_depth(x)
        # forloop ImageExtractor
        for i in range(self.down_step):
            y = self.DList[i](y)

        return y
