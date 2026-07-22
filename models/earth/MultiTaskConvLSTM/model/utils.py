#Definition of evaluation metrics
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
import torch
from scipy.stats import kendalltau
import scipy.stats as stats


def nash_sutcliffe_efficiency(observed, predicted):
    # Ensure inputs are tensors on the CPU
    observed = observed.cpu()
    predicted = predicted.cpu()

    # Compute the numerator and denominator
    numerator = torch.sum((observed - predicted) ** 2)
    denominator = torch.sum((observed - torch.mean(observed)) ** 2)

    # Calculate NSE
    nse = 1 - (numerator / denominator)
    return nse.item()

def pearson_correlation(y_true, y_pred):
    y_true = y_true.view(-1).cpu().numpy()  # Flatten and move to CPU
    y_pred = y_pred.view(-1).cpu().numpy()  # Flatten and move to CPU
    
    return pearsonr(y_true, y_pred)[0]  # Return the correlation coefficient

def spearman_correlation(y_true, y_pred):
    y_true = y_true.view(-1).cpu().numpy()  # Flatten and move to CPU
    y_pred = y_pred.view(-1).cpu().numpy()  # Flatten and move to CPU
    
    return spearmanr(y_true, y_pred).correlation  # Return the Spearman correlation

def mse(y_true, y_pred):
    # Ensure inputs are tensors on the CPU
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    
    return torch.mean((y_true - y_pred) ** 2).item()

def mae(y_true, y_pred):
    # Ensure inputs are tensors on the CPU
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    
    return torch.mean(torch.abs(y_true - y_pred)).item()

def percentage_error(y_true, y_pred):
    # Ensure inputs are tensors on the CPU
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    
    return 100 * torch.mean((y_pred - y_true) / (y_true + 1e-6)).item()

def percentage_bias(y_true, y_pred):
    # Ensure inputs are tensors on the CPU
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    return 100 * torch.sum(y_pred - y_true) / (torch.sum(y_true) + 1e-6)

def kendall_tau(y_true, y_pred):
    y_true = y_true.view(-1).cpu().numpy()  # Flatten and move to CPU
    y_pred = y_pred.view(-1).cpu().numpy()  # Flatten and move to CPU
    
    return kendalltau(y_true, y_pred).correlation  # Return the Kendall Tau

def r2_score(y_true, y_pred):
    # Ensure inputs are tensors on the CPU
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    
    return 1 - (ss_residual / (ss_total + 1e-6)).item()

def spatial_correlation(y_true, y_pred):
    # Flatten the tensors to work with them
    y_true_flat = y_true.view(-1).cpu()
    y_pred_flat = y_pred.view(-1).cpu()

    # Compute the numerator: sum(P * T)
    numerator = torch.sum(y_pred_flat * y_true_flat)

    # Compute the denominator: sqrt(sum(P^2) * sum(T^2))
    denominator = torch.sqrt(torch.sum(y_pred_flat ** 2) * torch.sum(y_true_flat ** 2))

    # Compute the correlation (add epsilon to avoid division by zero)
    correlation = numerator / (denominator)

    return correlation.item()