import os
import sys
import json
import yaml
import logging
import pathlib
import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from onescience.distributed.manager import DistributedManager
from RAE2822 import RAE2822Datapipe
from onescience.utils.YParams import YParams

from onescience.models.transolver import Transolver2D
from onescience.models.transolver import Transolver2D_plus
from onescience.models.transolver.MLP import MLP
from onescience.models.transolver.GraphSAGE import GraphSAGE
from onescience.models.transolver.PointNet import PointNet
from onescience.models.transolver.NN import NN
from onescience.models.transolver.GUNet import GUNet


def setup_logging():
    """日志初始化"""
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().setLevel(level)
    return logging.getLogger()


def main():
    DistributedManager.initialize()
    logger = setup_logging()

    # 配置加载
    config_file_path = "conf/transolver_rae2822.yaml"
    logger.info(f"Loading configuration from: {config_file_path}")

    cfg_model_all = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")

    # 设备选择
    use_cuda = 0 <= cfg_train.gpuid < torch.cuda.device_count() and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg_train.gpuid}' if use_cuda else 'cpu')
    logger.info(f"Running inference on device: {device}")

    # 模型参数
    model_name = cfg_model_all.name
    logger.info(f"Preparing to test model: {model_name}")
    if model_name not in cfg_model_all.specific_params:
        raise ValueError(f"Model '{model_name}' not found in config's 'specific_params' block.")
    model_params = cfg_model_all.specific_params[model_name]

    hparams = model_params
    hparams['subsampling'] = 32000

    # 数据管道初始化
    logger.info("Initializing datapipe...")
    cfg_data.model_hparams = model_params
    datapipe = RAE2822Datapipe(params=cfg_data, distributed=False)
    coef_norm = datapipe.coef_norm
    logger.info("Normalization coefficients loaded.")

    test_loader, _ = datapipe.test_dataloader()
    test_dataset_names = datapipe.test_dataset.data_list_names
    logger.info(f"Test loader initialized with {len(test_dataset_names)} samples.")

    # 模型构建
    logger.info(f"Initializing model architecture: {model_name}")
    if model_name in ['Transolver', 'Transolver_plus']:
        ModelClass = Transolver2D if model_name == 'Transolver' else Transolver2D_plus   
        model = ModelClass(
            n_hidden=model_params.n_hidden,
            n_layers=model_params.n_layers,
            space_dim=model_params.space_dim,
            fun_dim=model_params.fun_dim,
            n_head=model_params.n_head,
            mlp_ratio=model_params.mlp_ratio,
            out_dim=model_params.out_dim,
            slice_num=model_params.slice_num,
            unified_pos=model_params.unified_pos
        ).to(device)
    else:
        encoder = MLP(list(model_params.encoder), batch_norm=False)
        decoder = MLP(list(model_params.decoder), batch_norm=False)
        if model_name == 'GraphSAGE':
            model = GraphSAGE(hparams, encoder, decoder).to(device)
        elif model_name == 'PointNet':
            model = PointNet(hparams, encoder, decoder).to(device)
        elif model_name == 'MLP':
            model = NN(hparams, encoder, decoder).to(device)
        elif model_name == 'GUNet':
            model = GUNet(hparams, encoder, decoder).to(device)
        else:
            raise NotImplementedError(f"Model {model_name} initialization not implemented.")

    # 权重加载
    checkpoint_dir = cfg_train.checkpoint_dir
    model_path = osp.join(checkpoint_dir, f"{model_name}.pth")
    if not osp.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    logger.info(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 推理与指标计算
    logger.info("Starting inference...")
    path_out = osp.join(cfg_train.result_dir, 'rae2822')
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss(reduction='none')

    all_losses = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            out = model(data)
            targets = data.y

            # 计算损失
            loss = criterion(out, targets).mean().item()
            all_losses.append(loss)

            # 保存预测和目标
            all_predictions.append(out.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # 结果汇总
    avg_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算每个输出通道的误差
    channel_losses = []
    for i in range(all_predictions.shape[-1]):
        channel_loss = np.mean((all_predictions[..., i] - all_targets[..., i]) ** 2)
        channel_losses.append(channel_loss)

    # 保存结果
    score_file = osp.join(path_out, f'score_{model_name}.json')
    logger.info(f"Saving score summary to: {score_file}")
    with open(score_file, 'w') as f:
        json.dump(
            {
                'model_name': model_name,
                'mean_loss': float(avg_loss),
                'std_loss': float(std_loss),
                'channel_losses': [float(l) for l in channel_losses],
                'n_samples': len(test_dataset_names)
            },
            f, indent=4
        )

    # 保存预测和目标
    np.save(osp.join(path_out, 'predictions.npy'), all_predictions)
    np.save(osp.join(path_out, 'targets.npy'), all_targets)
    
    # 保存归一化系数（分别保存）
    if coef_norm is not None:
        mean_in, std_in, mean_out, std_out = coef_norm
        np.save(osp.join(path_out, 'mean_in.npy'), mean_in)
        np.save(osp.join(path_out, 'std_in.npy'), std_in)
        np.save(osp.join(path_out, 'mean_out.npy'), mean_out)
        np.save(osp.join(path_out, 'std_out.npy'), std_out)

    logger.info(f"=====  Inference and testing complete. =====")
    logger.info(f"Average Loss: {avg_loss:.6f} ± {std_loss:.6f}")
    logger.info(f"Channel Losses: {channel_losses}")


if __name__ == "__main__":
    main()