import torch
import torch.nn as nn
import torch.nn.functional as F
from onescience.modules import FuxiEmbedding, FuxiTransformer, FuxiFC
from config import FuXiConfig


class FuXi(nn.Module):
    def __init__(self, config: FuXiConfig):
        super().__init__()
        self.config = config
        
        self.embedding = FuxiEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim
        )
        
        self.transformer = FuxiTransformer(
            embed_dim=config.embed_dim,
            num_groups=config.num_groups,
            input_resolution=config.input_resolution,
            num_heads=config.num_heads,
            window_size=config.window_size,
            depth=config.depth
        )
        
        self.fc = FuxiFC(
            in_channels=config.embed_dim,
            out_channels=config.out_channels
        )
    
    def forward(self, x):
        B, C, T, Lat, Lon = x.shape
        
        x = self.embedding(x)
        x = x.squeeze(2)
        
        x = self.transformer(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc(x)
        
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        x = x.reshape(B, self.config.in_chans, 4, 4, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, self.config.in_chans, H * 4, W * 4)
        
        if Lat % 4 != 0:
            x = F.interpolate(x, size=(Lat, Lon), mode='bilinear', align_corners=False)
        
        return x
