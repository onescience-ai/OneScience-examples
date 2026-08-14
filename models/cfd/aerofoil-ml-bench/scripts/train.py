import os
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0001, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        if val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        return False


def get_nb_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, train_loader, optimizer, scheduler, criterion='MSE', reg=1, mat_sz=5):
    model.train()
    avg_loss_per_var = torch.zeros(mat_sz, device=device)
    avg_loss = 0
    avg_loss_surf_var = torch.zeros(mat_sz, device=device)
    avg_loss_vol_var = torch.zeros(mat_sz, device=device)
    avg_loss_surf = 0
    avg_loss_vol = 0
    it = 0
    for data in train_loader:
        data_clone = data.clone().to(device)
        optimizer.zero_grad()
        out = model(data_clone)
        targets = data_clone.y

        if criterion in ('MSE', 'MSE_weighted'):
            loss_criterion = nn.MSELoss(reduction='none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction='none')

        loss_per_var = loss_criterion(out, targets).mean(dim=0)
        total_loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf], targets[data_clone.surf]).mean(dim=0)
        loss_vol_var = loss_criterion(out[~data_clone.surf], targets[~data_clone.surf]).mean(dim=0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        if criterion == 'MSE_weighted':
            L = (loss_vol + reg * loss_surf)
        else:
            L = total_loss

        L.backward()
        optimizer.step()
        scheduler.step()

        avg_loss_per_var += loss_per_var
        avg_loss += total_loss
        avg_loss_surf_var += loss_surf_var
        avg_loss_vol_var += loss_vol_var
        avg_loss_surf += loss_surf
        avg_loss_vol += loss_vol
        it += 1

    return (avg_loss.detach().cpu().numpy() / it, avg_loss_per_var.detach().cpu().numpy() / it,
            avg_loss_surf_var.detach().cpu().numpy() / it, avg_loss_vol_var.detach().cpu().numpy() / it,
            avg_loss_surf.detach().cpu().numpy() / it, avg_loss_vol.detach().cpu().numpy() / it)


@torch.no_grad()
def test(device, model, test_loader, criterion='MSE', mat_sz=5):
    model.eval()
    final_outs = []
    avg_loss_per_var = np.zeros(mat_sz)
    avg_loss = 0
    avg_loss_surf_var = np.zeros(mat_sz)
    avg_loss_vol_var = np.zeros(mat_sz)
    avg_loss_surf = 0
    avg_loss_vol = 0
    it = 0
    tok = []
    for data in test_loader:
        data_clone = data.clone().to(device)
        tik = time.time()
        out = model(data_clone)
        tok.append(time.time() - tik)
        targets = data_clone.y

        if criterion in ('MSE', 'MSE_weighted'):
            loss_criterion = nn.MSELoss(reduction='none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction='none')

        loss_per_var = loss_criterion(out, targets).mean(dim=0)
        loss = loss_per_var.mean()
        loss_surf_var = loss_criterion(out[data_clone.surf], targets[data_clone.surf]).mean(dim=0)
        loss_vol_var = loss_criterion(out[~data_clone.surf], targets[~data_clone.surf]).mean(dim=0)
        loss_surf = loss_surf_var.mean()
        loss_vol = loss_vol_var.mean()

        avg_loss_per_var += loss_per_var.cpu().numpy()
        avg_loss += loss.cpu().numpy()
        avg_loss_surf_var += loss_surf_var.cpu().numpy()
        avg_loss_vol_var += loss_vol_var.cpu().numpy()
        avg_loss_surf += loss_surf.cpu().numpy()
        avg_loss_vol += loss_vol.cpu().numpy()

        data_clone.x = out
        data_outs = data_clone.to_data_list()
        final_outs.append(data_outs)
        it += 1

    return (final_outs, avg_loss / it, avg_loss_per_var / it,
            avg_loss_surf_var / it, avg_loss_vol_var / it,
            avg_loss_surf / it, avg_loss_vol / it, tok)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(device, train_dataset, val_dataset, Net, hparams, path, coef_norm,
         criterion='MSE', reg=1, val_iter=10, name_mod='GraphSAGE', val_sample=True,
         num_epochs=400):
    Path(path).mkdir(parents=True, exist_ok=True)
    model = Net.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * num_epochs,
    )
    early_stopping = EarlyStopping(patience=10, delta=0.0001)
    val_loader = DataLoader(val_dataset, batch_size=1)

    train_loss_list = []
    val_loss_list = []
    val_surf_list = []
    val_vol_list = []
    pbar = tqdm(range(num_epochs), desc='epochs')

    for epoch in pbar:
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            if name_mod not in ('PointNet', 'MLP'):
                data_sampled.edge_index = nng.radius_graph(
                    x=data_sampled.pos.to(device), r=hparams['r'], loop=True,
                    max_num_neighbors=int(hparams['max_neighbors'])
                ).cpu()
            train_dataset_sampled.append(data_sampled)
        train_loader = DataLoader(train_dataset_sampled, batch_size=hparams['batch_size'], shuffle=True)

        train_loss, _, loss_surf_var, loss_vol_var, loss_surf, loss_vol = train(
            device, model, train_loader, optimizer, lr_scheduler, criterion, reg=reg)
        if criterion == 'MSE_weighted':
            train_loss = reg * loss_surf + loss_vol
        train_loss_list.append(train_loss)

        if val_iter is not None and (epoch % val_iter == val_iter - 1 or epoch == 0):
            _, val_loss, _, _, _, val_surf, val_vol, _ = test(device, model, val_loader, criterion)
            val_loss_list.append(val_loss)
            val_surf_list.append(val_surf)
            val_vol_list.append(val_vol)
            if early_stopping(val_loss, model):
                print(f'Early stopping at epoch {epoch}')
                break

    model.load_state_dict(torch.load(early_stopping.path, map_location=device))
    return model
