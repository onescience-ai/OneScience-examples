import torch.nn as nn

from .DGMR_SO.model import Generator as DGMR_SO
from .Generator_only.model_clr_idx import Generator as Generator_only


class Predictor(nn.Module):
    def __init__(
        self,
        model_type,
    ):
        super().__init__()

        if model_type == 'DGMR_SO':
            self.generator = DGMR_SO(
                in_channels=1,
                base_channels=24,
                down_step=4,
                prev_step=4,
                sigma=1
            )

        elif model_type == 'Generator_only':
            self.generator = Generator_only(
                in_channels=1,
                base_channels=24,
                down_step=4,
                prev_step=4,
            )

    def forward(self, x, x2, topo, datetime_feat, pred_step=36):
        """
        x: input seq -> dims (N, D, C, H, W)
        x2: input seq (WRF) -> dims (N, D, C, H, W)
        topo: topography -> dims (N, 1, H=512, W=512)
        datetime_feat -> dims (N, D, 4)
        """
        pred = self.generator(x, x2, topo, datetime_feat, pred_step=pred_step)

        return pred
