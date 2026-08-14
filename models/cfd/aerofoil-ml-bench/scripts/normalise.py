import json
import os

import numpy as np
import torch


def fit(dataset):
    """Incrementally compute per-feature mean/std for x (input) and y (output)."""
    mean_in = None
    std_in = None
    mean_out = None
    std_out = None
    old_length = 0
    for k, data in enumerate(dataset):
        init = np.array(data.x)
        target = np.array(data.y)
        if k == 0:
            old_length = init.shape[0]
            mean_in = init.mean(axis=0, dtype=np.double)
            mean_out = target.mean(axis=0, dtype=np.double)
            std_in = ((init - mean_in) ** 2).sum(axis=0, dtype=np.double) / old_length
            std_out = ((target - mean_out) ** 2).sum(axis=0, dtype=np.double) / old_length
        else:
            new_length = old_length + init.shape[0]
            mean_in += (init.sum(axis=0, dtype=np.double) - init.shape[0] * mean_in) / new_length
            mean_out += (target.sum(axis=0, dtype=np.double) - init.shape[0] * mean_out) / new_length
            std_in += (((init - mean_in) ** 2).sum(axis=0, dtype=np.double) - init.shape[0] * std_in) / new_length
            std_out += (((target - mean_out) ** 2).sum(axis=0, dtype=np.double) - target.shape[0] * std_out) / new_length
            old_length = new_length

    std_in = np.sqrt(std_in).astype(np.single)
    std_out = np.sqrt(std_out).astype(np.single)
    coef_norm = [mean_in.astype(np.single), std_in, mean_out.astype(np.single), std_out]
    return coef_norm


def normalise(dataset, coeff_norm):
    mean_in, std_in, mean_out, std_out = coeff_norm
    for data in dataset:
        data.x = torch.tensor((np.array(data.x) - mean_in) / (np.array(std_in) + 1e-8)).to(torch.float32)
        data.y = torch.tensor((np.array(data.y) - mean_out) / (np.array(std_out) + 1e-8)).to(torch.float32)
    return dataset


def norm_data(data, coeff_norm):
    mean_in, std_in, mean_out, std_out = coeff_norm
    data.x = torch.tensor((np.array(data.x) - mean_in) / (np.array(std_in) + 1e-8)).to(torch.float32)
    data.y = torch.tensor((np.array(data.y) - mean_out) / (np.array(std_out) + 1e-8)).to(torch.float32)
    return data


def denormalise(dataset, coef_norm):
    mean_in, std_in, mean_out, std_out = coef_norm
    for data in dataset:
        data.x = torch.tensor(np.array(data.x) * (np.array(std_in) + 1e-8) + mean_in).to(torch.float32)
        data.y = torch.tensor(np.array(data.y) * (np.array(std_out) + 1e-8) + mean_out).to(torch.float32)
    return dataset


def denormalise_ys(dataset, coef_norm):
    mean_in, std_in, mean_out, std_out = coef_norm
    for l_data in dataset:
        data = l_data.cpu()
        data.x = torch.tensor(np.array(data.x) * (np.array(std_out) + 1e-8) + mean_out).to(torch.float32)
        data.y = torch.tensor(np.array(data.y) * (np.array(std_out) + 1e-8) + mean_out).to(torch.float32)
    return dataset


def load_normalisation_coefs(log_file_path=os.path.join('data', 'total_coeffs_log.json')):
    try:
        with open(log_file_path, 'r') as f:
            data = json.load(f)
            return data.get('normalisation_coefs', None)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_normalisation_coefs(coef_norm, log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    payload = {
        'normalisation_coefs': [
            coef_norm[0].tolist(), coef_norm[1].tolist(),
            coef_norm[2].tolist(), coef_norm[3].tolist(),
        ]
    }
    with open(log_file_path, 'w') as f:
        json.dump(payload, f)
