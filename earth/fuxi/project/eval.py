import torch
import numpy as np
from model import FuXi
from config import FuXiConfig


def calculate_rmse(pred, target, lat_dim=1):
    B, C, Lat, Lon = pred.shape
    lat_weights = torch.cos(torch.linspace(-np.pi/2, np.pi/2, Lat))
    lat_weights = lat_weights.to(pred.device)
    lat_weights = lat_weights / lat_weights.sum()
    
    mse = (pred - target) ** 2
    mse = mse * lat_weights[None, None, :, None]
    rmse = torch.sqrt(mse.mean())
    return rmse.item()


def calculate_acc(pred, target, climatology, lat_dim=1):
    B, C, Lat, Lon = pred.shape
    lat_weights = torch.cos(torch.linspace(-np.pi/2, np.pi/2, Lat))
    lat_weights = lat_weights.to(pred.device)
    lat_weights = lat_weights / lat_weights.sum()
    
    pred_anomaly = pred - climatology
    target_anomaly = target - climatology
    
    cov = (pred_anomaly * target_anomaly * lat_weights[None, None, :, None]).sum()
    pred_var = (pred_anomaly ** 2 * lat_weights[None, None, :, None]).sum()
    target_var = (target_anomaly ** 2 * lat_weights[None, None, :, None]).sum()
    
    acc = cov / torch.sqrt(pred_var * target_var + 1e-8)
    return acc.item()


def evaluate():
    config = FuXiConfig()
    device = config.device
    
    model = FuXi(config).to(device)
    model.load_state_dict(torch.load('fuxi_model.pth', map_location=device))
    model.eval()
    
    num_eval_samples = 10
    total_rmse = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for _ in range(num_eval_samples):
            x = torch.randn(
                1,
                config.in_chans,
                config.img_size[0],
                config.img_size[1],
                config.img_size[2]
            ).to(device)
            
            y = torch.randn(
                1,
                config.in_chans,
                config.img_size[1],
                config.img_size[2]
            ).to(device)
            
            climatology = torch.randn_like(y)
            
            pred = model(x)
            
            rmse = calculate_rmse(pred, y)
            acc = calculate_acc(pred, y, climatology)
            
            total_rmse += rmse
            total_acc += acc
    
    avg_rmse = total_rmse / num_eval_samples
    avg_acc = total_acc / num_eval_samples
    
    print(f'Evaluation Results:')
    print(f'  Average RMSE: {avg_rmse:.6f}')
    print(f'  Average ACC: {avg_acc:.6f}')


if __name__ == '__main__':
    evaluate()
