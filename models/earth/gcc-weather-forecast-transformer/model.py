#!/usr/bin/env python3
"""
Transformer model for weather forecasting.
Updated for 4 features: temperature, humidity, precipitation, wind speed.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class WeatherTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=256, n_heads=8, n_layers=4, 
                 d_ff=512, dropout=0.1, seq_len=72):
        super(WeatherTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.output_head(x)
        return out.squeeze(-1)


def create_model(device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
    model = WeatherTransformer(**kwargs)
    return model.to(device)


if __name__ == "__main__":
    print("🧪 Testing WeatherTransformer...\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    config = {
        'input_dim': 4,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'seq_len': 72
    }
    
    model = create_model(device=device, **config)
    
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")
    print(f"Size: {n_params * 4 / 1e6:.1f} MB")
    
    x = torch.randn(32, 72, 4).to(device)
    with torch.no_grad():
        out = model(x)
    
    print(f"\nInput: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"\n✅ Model ready!")
