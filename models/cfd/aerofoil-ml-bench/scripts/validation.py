import argparse
import os.path as osp

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import train
from normalise import fit, normalise
from data.dataset import make_dataset
from post_proc.panel_method import lift_coef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='MLP, GraphSAGE, PointNet, GUNet', type=str)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', default='metrics')
    parser.add_argument('--subsample', type=int, default=10000)
    parser.add_argument('--n-foils', type=int, default=10, help='training sample count (path)')
    parser.add_argument('--n-val', type=int, default=30, help='val samples used for norm stats')
    parser.add_argument('--n-test', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--graph', action='store_true')
    args = parser.parse_args()

    checkpoint = torch.load(osp.join(args.out_dir, f'{args.n_foils}_samples', args.model, 'model.pt'),
                            map_location='cpu', weights_only=False)
    model = checkpoint['model']
    coef_norm = checkpoint['coef_norm']
    model.eval()

    all_data = make_dataset(source_dir=args.data_dir, graph=args.graph,
                            subsample=args.subsample, seed=args.seed,
                            n_samples=args.n_val + args.n_test)
    # use first n_val for norm-fit consistency with training split
    norm_data = [d for d in all_data[:args.n_val]]
    test_data = [d for d in all_data[args.n_val:args.n_val + args.n_test]]

    _coef = fit(norm_data)
    test_norm = normalise(test_data, _coef)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loader = DataLoader(test_norm, batch_size=1)

    final_outs, tloss, tpervar, tsurfvar, tvolvar, tsurf, tvol, tok = train.test(
        device, model, loader, criterion='MSE', mat_sz=5)

    print('Test RMSE total:', np.sqrt(tloss))
    print('Test RMSE surface:', np.sqrt(tsurf))
    print('Test RMSE fluid:', np.sqrt(tvol))
    print('Test RMSE per-var:', np.sqrt(tpervar))
    print('Inference time per sample (s):', np.mean(tok))

    try:
        cl_rmse, cl_mse, out_list = lift_coef(final_outs, _coef)
        print('CL RMSE:', cl_rmse)
    except Exception as e:
        print('CL calc skipped:', e)


if __name__ == '__main__':
    main()
