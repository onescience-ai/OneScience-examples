import torch.nn as nn

from .generator_clr_idx_wrf_topot import Sampler, ContextConditionStack


class Generator(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels,
        down_step,
        prev_step
    ):

        super().__init__()
        self.contextStack = ContextConditionStack(
            in_channels=in_channels,
            base_channels=base_channels,
            down_step=down_step,
            prev_step=prev_step
        )

        self.sampler = Sampler(
            in_channels=in_channels,
            base_channels=base_channels,
            up_step=down_step,
        )

    def forward(
        self,
        x,
        x2,
        topo,
        time_feat,
        pred_step=12,
        y=None,
        thres=None
    ):
        """
        x: input seq -> dims (N, T, C, H, W)
        x2: input seq for WRF -> dims (N, T, C, H, W)
        """
        context_inits = self.contextStack(x)
        if y is None:
            y = x[:, -1:, :, :, :]

        pred = self.sampler(
            y,
            x2,
            topo,
            time_feat,
            context_inits,
            pred_step,
            thres
        )

        return pred
