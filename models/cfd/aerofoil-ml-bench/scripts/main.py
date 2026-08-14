import argparse
import os
import os.path as osp
import random

import torch
from tqdm import tqdm

import train
from normalise import fit, normalise
from data.dataset import make_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='MLP, GraphSAGE, PointNet, GUNet', type=str)
    parser.add_argument('--data-dir', required=True, help='dir with NACA_Cylinder_{X,Y,Q}.npy')
    parser.add_argument('--out-dir', default='metrics', help='output dir')
    parser.add_argument('--foils', type=int, default=5, help='number of training samples')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--subsample', type=int, default=10000, help='max nodes per sample')
    parser.add_argument('--n-total', type=int, default=120, help='total samples to load')
    parser.add_argument('--n-val', type=int, default=30, help='validation samples')
    parser.add_argument('--n-test', type=int, default=30, help='test samples')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--graph', action='store_true', help='build edge_index')
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load all samples and split into train/val/test (foils stand for "samples")
    all_data = make_dataset(
        source_dir=args.data_dir, graph=args.graph,
        subsample=args.subsample, seed=args.seed, n_samples=args.n_total,
    )
    rng = random.Random(args.seed)
    idx = list(range(len(all_data)))
    rng.shuffle(idx)
    n_train = args.foils
    n_val = args.n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:n_train + n_val + args.n_test]

    train_dataset = [all_data[i] for i in train_idx]
    val_dataset = [all_data[i] for i in val_idx]
    test_dataset = [all_data[i] for i in test_idx]

    coef_norm = fit(train_dataset)
    train_dataset = normalise(train_dataset, coef_norm)
    val_dataset = normalise(val_dataset, coef_norm)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    with open('params.yaml', 'r') as f:
        import yaml
        hparams = yaml.safe_load(f)[args.model]

    from models.MLP import MLP
    encoder = MLP(channel_list=hparams['encoder'], batch_norm=False)
    decoder = MLP(channel_list=hparams['decoder'], batch_norm=False)

    if args.model == 'GraphSAGE':
        from models.GraphSAGE import GraphSAGE as Net
    elif args.model == 'PointNet':
        from models.PointNet import PointNet as Net
    elif args.model == 'MLP':
        from models.NN import NN as Net
    elif args.model == 'GUNet':
        from models.GUNet import GUNet as Net
    else:
        raise ValueError(f'Unknown model {args.model}')

    model = Net(hparams, encoder, decoder)

    log_path = osp.join(args.out_dir, f'{args.foils}_samples', args.model)
    model = train.main(device, train_dataset, val_dataset, model, hparams, log_path,
                       coef_norm, criterion='MSE', val_iter=10, name_mod=args.model,
                       val_sample=True, num_epochs=args.epochs)

    torch.save({'model': model, 'coef_norm': coef_norm,
                'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx},
               osp.join(log_path, 'model.pt'))
    print('Saved model to', osp.join(log_path, 'model.pt'))
    print('Test sample indices:', test_idx)


if __name__ == '__main__':
    main()
