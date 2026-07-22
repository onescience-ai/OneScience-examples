from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import einops

from ..components.ConvGRU import ConvGRUCell
from ..components.common import GBlock, Up_GBlock, DBlock


class Sampler(nn.Module):
    def __init__(self, in_channels, base_channels=24, up_step=4):
        """
        up_step should be the same as down_step in context-condition-stack

        """
        super().__init__()
        base_c = base_channels

        self.up_steps = up_step
        self.convgru_list = nn.ModuleList()
        self.conv1x1_list = nn.ModuleList()
        self.gblock_list = nn.ModuleList()
        self.upg_list = nn.ModuleList()

        # image extractor
        self.img_extractor = ImageExtractor(
            in_channels=2,
            out_channels=base_c * 2**(self.up_steps) * in_channels,
            apply_down_flag=[True, True, True, True],
            down_step=self.up_steps
        )

        self.nwp_extractor = ImageExtractor(
            in_channels=1,
            out_channels=base_c * 2**(self.up_steps) * in_channels,
            apply_down_flag=[False, True, False, True],
            down_step=self.up_steps
        )

        self.encode_time = nn.Linear(
            4, base_c * 2**(self.up_steps + 1) * in_channels)

        for i in range(self.up_steps):
            # different scale
            chs1 = base_c * 2**(self.up_steps - i + 1) * in_channels
            chs2 = base_c * 2**(self.up_steps - i) * in_channels
            # convgru
            self.convgru_list.append(
                ConvGRUCell(chs1, chs2, 3)
            )
            # conv1x1
            self.conv1x1_list.append(
                spectral_norm(
                    nn.Conv2d(
                        in_channels=chs2,
                        out_channels=chs1,
                        kernel_size=(
                            1,
                            1))))
            # GBlock
            self.gblock_list.append(
                GBlock(in_channel=chs1, out_channel=chs1)
            )
            # upgblock
            self.upg_list.append(
                Up_GBlock(in_channel=chs1)
            )

            # output
            # self.bn = nn.BatchNorm2d(chs2)
            self.relu = nn.ReLU()
            self.last_conv1x1 = spectral_norm(
                nn.Conv2d(in_channels=chs2,
                          out_channels=4,
                          kernel_size=(1, 1))
            )
            self.depth_to_space = nn.PixelShuffle(upscale_factor=2)

    def forward(
            self,
            input_img,
            nwp_inputs,
            topo,
            time_feat,
            init_states,
            pred_step,
            thres=None):
        """
        input_img dim  -> (N, Tstep, C, W, H) -> Tstep can be 1 or pred_steps
        nwp_inputs dim -> (N, Tstep, C, W, H)
        init_states dim -> [(N, C, W, H)-1, (N, C, W, H)-2, ...]
        probs -> (tsteps)
        """
        hh = init_states
        output = []
        img_t_len = input_img.shape[1]

        xx = None
        for t in range(pred_step):
            time_emb = self.encode_time(time_feat[:, t])  # time emb

            # The 1st tstep image should be truth
            # and 2nd tstep image apply schedule sampling
            if t == 0:
                input_ = input_img[:, 0, :, :, :]
            else:
                if thres is not None and img_t_len > 1:
                    # use groud truth
                    if torch.rand(1) < thres:
                        input_ = input_img[:, t, :, :, :]
                    else:
                        input_ = xx
                else:
                    input_ = xx

            nwp_in = nwp_inputs[:, t]
            # image extractor -> extract T step image
            # xx is the output of image extractor
            input_ = torch.cat((input_, topo), dim=1)
            xx = self.img_extractor(input_)
            nwp_feat = self.nwp_extractor(nwp_in)
            xx = torch.cat((xx, nwp_feat), dim=1)

            xx = xx + time_emb[:, :, None, None]  # add time embedding

            for up in range(self.up_steps):
                # convGRU
                # init_states should be reversed
                h_index = (self.up_steps - 1) - up
                xx, out_hh = self.convgru_list[up](xx, hh[h_index])
                hh[h_index] = out_hh
                # conv1x1
                xx = self.conv1x1_list[up](xx)
                # gblock
                xx = self.gblock_list[up](xx)
                # upg_list
                xx = self.upg_list[up](xx)

            # xx = self.bn(xx)
            xx = self.relu(xx)
            xx = self.last_conv1x1(xx)
            xx = self.depth_to_space(xx)

            # prediction
            output.append(xx)

        output = torch.stack(output, dim=1)

        return output


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
        y = self.space_to_depth(x)
        # forloop ImageExtractor
        for i in range(self.down_step):
            y = self.DList[i](y)

        return y


class ContextConditionStack(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 24,
                 down_step: int = 4,
                 prev_step: int = 4):
        """
        base_channels: e.g. 24 -> output_channel: 384
        output_channel: base_c*in_c*2**(down_step-2) * prev_step
        down_step: int
        prev_step: int
        """
        super().__init__()
        self.in_channels = in_channels
        self.down_step = down_step
        self.prev_step = prev_step
        ###
        base_c = base_channels
        in_c = in_channels

        # different scales channels
        chs = [4 * in_c] + [base_c * in_c * 2 **
                            (i + 1) for i in range(down_step)]

        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=2)
        self.Dlist = nn.ModuleList()
        self.convList = nn.ModuleList()
        for i in range(down_step):
            self.Dlist.append(
                DBlock(in_channel=chs[i],
                       out_channel=chs[i + 1],
                       apply_relu=True, apply_down=True)
            )

            self.convList.append(
                spectral_norm(
                    nn.Conv2d(in_channels=prev_step * chs[i + 1],
                              out_channels=prev_step * chs[i + 1] // 4,
                              kernel_size=(3, 3),
                              padding=1)
                )
            )

        # ReLU
        self.relu = nn.ReLU()

    def forward(self,
                x: torch.Tensor) -> Tuple[torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor]:
        """
        ## input dims -> (N, D, C, H, W)
        """
        x = self.space_to_depth(x)
        tsteps = x.shape[1]
        assert tsteps == self.prev_step

        # different feature index represent different scale
        # features
        # [scale1 -> [t1, t2, t3, t4], scale2 -> [t1, t2, t3, t4], scale3 -> [....]]
        features = [[] for i in range(tsteps)]

        for st in range(tsteps):
            in_x = x[:, st, :, :, :]
            # in_x -> (Batch(N), C, H, W)
            for scale in range(self.down_step):
                in_x = self.Dlist[scale](in_x)
                features[scale].append(in_x)

        out_scale = []
        for i, cc in enumerate(self.convList):
            # after stacking, dims -> (Batch, Time, C, H, W)
            # and mixing layer is to concat Time, C
            stacked = self._mixing_layer(torch.stack(features[i], dim=1))
            out = self.relu(cc(stacked))
            out_scale.append(out)

        return out_scale

    def _mixing_layer(self, x):
        # conver from (N, Time, C, H, W) -> (N, Time*C, H, W)
        # Then apply Conv2d
        stacked = einops.rearrange(x, "b t c h w -> b (t c) h w")

        return stacked
