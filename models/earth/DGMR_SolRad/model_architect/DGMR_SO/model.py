import torch
import torch.nn as nn
from .discriminator import TemporalDiscriminator, SpatialDiscriminator
from .generator import Sampler, ContextConditionStack, LatentConditionStack
from .img_extractor import ImageExtractor


class Generator(nn.Module):
    def __init__(
            self,
            in_channels,
            base_channels,
            down_step,
            prev_step,
            sigma
    ):
        super().__init__()
        out_channels = base_channels * \
            2**(down_step - 2) * prev_step * in_channels

        self.latentStack = LatentConditionStack(
            out_channels=out_channels,
            down_step=down_step,
            sigma=sigma
        )

        self.contextStack = ContextConditionStack(
            in_channels=in_channels,
            base_channels=base_channels,
            down_step=down_step,
            prev_step=prev_step
        )

        self.sampler = Sampler(
            in_channels=in_channels,
            base_channels=base_channels,
            up_step=down_step
        )

        self.encode_time = nn.Linear(
            4, base_channels * 2**(down_step) * in_channels)

        self.topo_extractor = ImageExtractor(
            in_channels=1,
            out_channels=base_channels * 2**(down_step - 1) * in_channels,
            apply_down_flag=[True, True, True, True],
            down_step=down_step
        )

        self.nwp_extractor = ImageExtractor(
            in_channels=1,  # TODO: fixed now
            out_channels=base_channels * 2**(down_step - 1) * in_channels,
            apply_down_flag=[False, True, False, True],
            down_step=down_step
        )

    def forward(self, x, x2, topo, datetime_feat, pred_step=36):
        """
        x: input seq -> dims (N, D, C, H, W)
        x2: input seq (WRF) -> dims (N, D, C, H, W)
        topo: topography -> dims (N, 1, H=512, W=512)
        datetime_feat -> dims (N, D, 4)
        """
        context_inits = self.contextStack(x)
        batch_size = context_inits[0].shape[0]
        zlatent = self.latentStack(x, batch_size=batch_size)

        # topo feature
        topo_feat = self.topo_extractor(topo)
        # encode time feature
        time_feat = self.encode_time(datetime_feat)
        # extract nwp feature
        nwp_feat = []
        # forloop x2
        for i in range(x2.shape[1]):
            nwp_ = self.nwp_extractor(x2[:, i, ...])
            # concat topo and nwp feature
            concat_feat = torch.cat((nwp_, topo_feat), dim=1)
            nwp_feat.append(concat_feat)
        nwp_feat = torch.stack(nwp_feat, dim=1)
        fuse_feat = nwp_feat + time_feat.unsqueeze(-1).unsqueeze(-1)

        pred = self.sampler(zlatent, fuse_feat, context_inits, pred_step)

        return pred


class Discriminator(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.spatial = SpatialDiscriminator(
            in_channel=in_channels, base_c=base_channels)
        self.temporal = TemporalDiscriminator(
            in_channel=in_channels, base_c=base_channels)

    def forward(self, x, y):
        """
        x -> dims (N, D, C, H, W) e.g. input_frames
        y -> dims (N, D, C, H, W) e.g. output_grames
        """
        spatial_out = self.spatial(y)
        temporal_out = self.temporal(torch.cat([x, y], dim=1))

        dis_out = torch.cat([spatial_out, temporal_out], dim=1)

        return dis_out
