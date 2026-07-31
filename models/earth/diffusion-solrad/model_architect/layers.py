import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_init import default_init

class SPADE(nn.Module):
    def __init__(self, norm_nc, cond_nc, spade_dim=128, param_free_norm_type='group'):
        """
        SPADE (Spatially Adaptive Normalization) layer.
        norm_nc: number of channels of the normalized feature map
        cond_nc: number of channels of the conditional map
        """
        super().__init__()

        if param_free_norm_type == 'group':
            num_groups = min(norm_nc // 4, 32)
            while(norm_nc % num_groups != 0): # must find another value
                num_groups -= 1
            self.param_free_norm = nn.GroupNorm(num_groups=num_groups, num_channels=norm_nc, affine=False, eps=1e-6)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_nc, spade_dim, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(spade_dim, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(spade_dim, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, cond_map):
        ## do param-free normalization (GroupNorm / InstanceNorm / BatchNorm)
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        cond_map = F.interpolate(cond_map, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(cond_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class ActNorm(nn.Module):
  def __init__(self, emb_dim, out_dim):
    super(ActNorm, self).__init__()

    ## For Time embedding
    chs = 2 * out_dim
    self.fc = nn.Linear(emb_dim, chs)
    self.fc.weight.data = default_init()(self.fc.weight.shape)
    nn.init.zeros_(self.fc.bias)

    self.activation = nn.SiLU()

  def forward(self, x, t_emb):
    """
    x: dim(B, C, H, W) or dim(B, C*N, H, W) if 3D
    t_emb: dim(B, emb_dim)
    """
    # ada-norm as in https://github.com/openai/guided-diffusion
    emb = self.activation(t_emb)
    emb_out = self.fc(emb)[:, :, None, None] # Linear projection
    scale, shift = torch.chunk(emb_out, 2, dim=1)

    y = x * (1 + scale) + shift

    return y
  
class Upsample_with_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)

    def forward(self, x):
        y = self.up(x)
        y = self.conv(y)

        return y

class Downsample_with_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y
    

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        cond_nc,
        emb_dim,
        spade_dim=128,
        dropout=0.1,
        param_free_norm_type='group',
        up_flag=False,
        down_flag=False
    ):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.cond_nc = cond_nc
        self.emb_dim = emb_dim
        self.up_flag = up_flag
        self.down_flag = down_flag

        self.activation = nn.SiLU()

        ## first
        self.spade1 = SPADE(in_c, cond_nc, spade_dim, param_free_norm_type)
        self.act_norm1 = ActNorm(emb_dim, in_c)
        self.conv1 = nn.Conv2d(in_c, in_c, 3, padding=1)
        
        ## downsampling or upsampling
        if up_flag:
            self.up_or_down_layer = Upsample_with_conv(in_c, out_c)
            self.skip_layer = nn.Upsample(scale_factor=2, mode="nearest")
        elif down_flag:
            self.up_or_down_layer = Downsample_with_conv(in_c, out_c)
            self.skip_layer = nn.AvgPool2d(2)
        else:
            self.conv_no_change = nn.Conv2d(in_c, out_c, 3, padding=1)
        
        ## second
        self.spade2 = SPADE(out_c, cond_nc, spade_dim, param_free_norm_type)
        self.act_norm2 = ActNorm(emb_dim, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)

        self.dropout = nn.Dropout(dropout)
        ## skip connection
        if in_c != out_c:
            self.conv1x1 = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x, cond, t_emb):
        """
        x: dim(B, C, H, W) or dim(B, C*N, H, W) if 3D
        cond: dim(B, cond_nc, H_cond, W_cond)
        t_emb: dim(B, emb_dim)
        """
        h = x
        ## first
        h = self.spade1(h, cond)
        h = self.act_norm1(h, t_emb)
        h = self.activation(h)
        h = self.conv1(h)
        
        ## up or down
        if self.up_flag or self.down_flag:
            x = self.skip_layer(x)
            h = self.up_or_down_layer(h)
        else:
            h = self.conv_no_change(h)

        ## second
        h = self.spade2(h, cond)
        h = self.act_norm2(h, t_emb)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        ## skip connection
        if self.in_c != self.out_c:
            x = self.conv1x1(x)

        return x + h
    
class AttnBlock(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.output_layer = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x):
        batch, channel, height, width = x.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(x)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, -1)
        query, key, value = qkv.chunk(3, dim=2)  # b, n_head, head_dim, h*w

        attn = torch.einsum(
            "bndL, bndM -> bnLM", query, key
        ).contiguous() / math.sqrt(head_dim)
        attn = torch.softmax(attn, -1)
        out = torch.einsum("bnLM, bndM -> bndL", attn, value).contiguous()
        out = out.view(batch, channel, height, width)
        out = self.output_layer(out)

        return out + x
    
def CropNConcat(x1, x2):
    row_diff = x2.shape[3] - x1.shape[3]
    col_diff = x2.shape[2] - x1.shape[2]

    x1 = F.pad(x1, [row_diff // 2, row_diff - row_diff // 2,
                     col_diff // 2, col_diff - col_diff // 2])

    out = torch.cat([x1, x2], dim=1)

    return out
