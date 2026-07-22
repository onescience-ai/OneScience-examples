import torch
import numpy as np
import argparse

from model_architect.inference_model import Predictor


def data_loading(BASETIME, device):
    data_npz = np.load(f'./sample_data/sample_{BASETIME}.npz')

    inputs = {}
    for key in data_npz:
        inputs[key] = torch.from_numpy(data_npz[key]).to(device)

    return inputs


def model_loading(model_type, device):
    if model_type == 'DGMR_SO':
        ckpt_path = './model_weights/DGMR_SO/ft36/weights.ckpt'
    elif model_type == 'Generator_only':
        ckpt_path = './model_weights/Generator_only/ft36/weights.ckpt'

    model = Predictor(
        model_type=model_type,
    )

    ckpt = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(ckpt['generator_state_dict'])
    model.eval()
    model.to(device)

    return model


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-type',
        type=str,
        default='DGMR_SO',
        choices=[
            'Generator_only',
            'DGMR_SO'])
    parser.add_argument('--basetime', type=str, default='202504131100')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    model_type = args.model_type
    BASETIME = args.basetime
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inputs = data_loading(BASETIME, device)
    model = model_loading(model_type, device)

    # prediction
    with torch.no_grad():
        pred_clr_idx = model(
            inputs['Himawari'],
            inputs['WRF'],
            inputs['topo'],
            inputs['time_feat'],
            pred_step=36,
        )
    pred_clr_idx = pred_clr_idx.squeeze(2).clamp(0, 1)

    # transform clearsky index to solar radiation
    pred_srad = pred_clr_idx * inputs['clearsky']  # dim: (1, 36, 512, 512)

    # save prediction
    np.save(f'./pred_{BASETIME}_{model_type}.npy', pred_srad.cpu().numpy())
    print('Done')
