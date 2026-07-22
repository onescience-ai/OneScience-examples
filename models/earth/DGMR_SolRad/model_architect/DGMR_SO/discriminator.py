import torch
import torch.nn as nn
from ..components.common import DBlock
import einops


class TemporalDiscriminator(nn.Module):
    def __init__(self, in_channel: int, base_c: int = 24):
        super().__init__()
        self.in_channel = in_channel

        # (N, C, D, H, W)
        self.down_sample = nn.AvgPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )

        # (N, D, C, H, W)
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=2)

        # in_channel, out_channel
        # Conv3D -> (N, C, D, H, W)
        chn = base_c * 2 * in_channel
        self.d3_1 = DBlock(in_channel=in_channel * 4,
                           out_channel=chn,
                           conv_type='3d', apply_relu=False, apply_down=True)

        self.d3_2 = DBlock(in_channel=chn,
                           out_channel=2 * chn,
                           conv_type='3d', apply_relu=True, apply_down=True)

        self.Dlist = nn.ModuleList()
        for i in range(3):
            chn = chn * 2
            self.Dlist.append(
                DBlock(in_channel=chn,
                       out_channel=2 * chn,
                       conv_type='2d', apply_relu=True, apply_down=True)
            )

        self.last_D = DBlock(in_channel=2 * chn,
                             out_channel=2 * chn,
                             conv_type='2d', apply_relu=True, apply_down=False)

        self.fc = nn.Linear(2 * chn, 1)
        self.relu = nn.ReLU()
        # TODO: close bn
        # self.bn = nn.BatchNorm1d(2*chn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_sample(x)
        x = self.space_to_depth(x)

        # go through the 3D Block
        # from (N, D, C, H, W) -> (N, C, D, H, W)
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        x = self.d3_1(x)
        x = self.d3_2(x)
        # go through 2D Block, permute -> (N, D, C, H, W)
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        n, d, c, h, w = list(x.size())
        ####
        fea = einops.rearrange(x, "n d c h w -> (n d) c h w")
        for dd in self.Dlist:
            fea = dd(fea)

        fea = self.last_D(fea)

        fea = torch.sum(self.relu(fea), dim=[2, 3])
        # fea = self.bn(fea)
        fea = self.fc(fea)

        y = torch.reshape(fea, (n, d, 1))  # dims -> (N, D, 1)
        y = torch.sum(y, keepdim=True, dim=1)  # dims -> (N, 1, 1)

        return y


class SpatialDiscriminator(nn.Module):
    def __init__(self, in_channel: int, base_c: int = 24):
        super().__init__()
        self.in_channel = in_channel

        # (N, C, H, W)
        self.down_sample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        # (N, C, H, W)
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=2)

        # first Dblock doesn't apply relu
        chn = base_c * in_channel
        self.d1 = DBlock(in_channel=in_channel * 4,
                         out_channel=chn * 2,
                         conv_type='2d', apply_relu=False, apply_down=True)

        self.Dlist = nn.ModuleList()
        for i in range(4):
            chn = chn * 2
            self.Dlist.append(
                DBlock(in_channel=chn,
                       out_channel=2 * chn,
                       conv_type='2d', apply_relu=True, apply_down=True)
            )

        self.last_D = DBlock(in_channel=2 * chn,
                             out_channel=2 * chn,
                             conv_type='2d', apply_relu=True, apply_down=False)

        self.fc = nn.Linear(2 * chn, 1)
        self.relu = nn.ReLU()
        # TODO: close BN
        # self.bn = nn.BatchNorm1d(2*chn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # note: input dims -> (N, D, C, H, W)
        # randomly pick up 8 out of 18
        perm = torch.randperm(x.shape[1])
        random_idx = perm[:8]

        fea = x[:, random_idx, :, :, :]

        n, d, c, h, w = list(fea.size())

        fea = einops.rearrange(fea, "n d c h w -> (n d) c h w")
        fea = self.down_sample(fea)
        fea = self.space_to_depth(fea)

        # apply DBlock
        fea = self.d1(fea)
        for dd in self.Dlist:
            fea = dd(fea)

        fea = self.last_D(fea)

        # sum
        fea = torch.sum(self.relu(fea), dim=[2, 3])
        # fea = self.bn(fea)
        fea = self.fc(fea)

        y = torch.reshape(fea, (n, d, 1))  # dims -> (N, D, 1)
        y = torch.sum(y, keepdim=True, dim=1)  # dims -> (N, 1, 1)

        return y
