import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import FuXi
from config import FuXiConfig
import numpy as np


class WeatherDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.config = FuXiConfig()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = torch.randn(
            self.config.in_chans,
            self.config.img_size[0],
            self.config.img_size[1],
            self.config.img_size[2]
        )
        y = torch.randn(
            self.config.in_chans,
            self.config.img_size[1],
            self.config.img_size[2]
        )
        return x, y


def latitude_weighted_l1_loss(pred, target, lat_dim=1):
    B, C, Lat, Lon = pred.shape
    lat_weights = torch.cos(torch.linspace(-np.pi/2, np.pi/2, Lat))
    lat_weights = lat_weights.to(pred.device)
    lat_weights = lat_weights / lat_weights.sum()
    
    loss = torch.abs(pred - target)
    loss = loss * lat_weights[None, None, :, None]
    loss = loss.mean()
    return loss


def train():
    config = FuXiConfig()
    device = config.device
    
    model = FuXi(config).to(device)
    
    dataset = WeatherDataset()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            pred = model(x)
            loss = latitude_weighted_l1_loss(pred, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.6f}')
    
    torch.save(model.state_dict(), 'fuxi_model.pth')
    print('Training completed. Model saved as fuxi_model.pth')


if __name__ == '__main__':
    train()
