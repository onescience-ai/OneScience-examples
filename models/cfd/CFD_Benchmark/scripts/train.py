import os
from pathlib import Path
from types import SimpleNamespace

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
torch = None
get_data = None
DistributedManager = None
replace_function = None
get_model = None
DerivLoss = None
L2Loss = None


def load_config():
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_onescience():
    global torch
    global get_data
    global DistributedManager
    global replace_function
    global get_model
    global DerivLoss
    global L2Loss

    if torch is not None:
        return

    import torch as torch_module
    from onescience.datapipes.cfd_benchmark.data_factory import get_data as get_data_fn
    from onescience.distributed.manager import DistributedManager as DistributedManager_cls
    from onescience.memory.checkpoint import replace_function as replace_function_fn
    from onescience.models.cfd_benchmark.model_factory import get_model as get_model_fn
    from onescience.utils.cfd_benchmark.loss import DerivLoss as DerivLoss_cls
    from onescience.utils.cfd_benchmark.loss import L2Loss as L2Loss_cls

    torch = torch_module
    get_data = get_data_fn
    DistributedManager = DistributedManager_cls
    replace_function = replace_function_fn
    get_model = get_model_fn
    DerivLoss = DerivLoss_cls
    L2Loss = L2Loss_cls


def build_args(cfg):
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["train"]
    return SimpleNamespace(
        lr=train["lr"],
        epochs=train["epochs"],
        weight_decay=train["weight_decay"],
        pct_start=train["pct_start"],
        batch_size=data["batch_size"],
        gpu=0,
        max_grad_norm=train["max_grad_norm"],
        derivloss=train["derivloss"],
        optimizer=train["optimizer"],
        scheduler=train["scheduler"],
        step_size=100,
        gamma=0.5,
        find_unused_parameters=False,
        use_checkpoint=train["use_checkpoint"],
        checkpoint_layers=train["checkpoint_layers"],
        resume=False,
        data_path=cfg["paths"]["data_dir"],
        loader=data["loader"],
        config_name="config_name",
        train_ratio=0.8,
        ntrain=data["ntrain"],
        ntest=data["ntest"],
        normalize=data["normalize"],
        norm_type=data["norm_type"],
        geotype=data["geotype"],
        time_input=False,
        space_dim=data["space_dim"],
        fun_dim=data["fun_dim"],
        out_dim=data["out_dim"],
        shapelist=None,
        downsamplex=data["downsamplex"],
        downsampley=data["downsampley"],
        downsamplez=1,
        radius=0.2,
        task=data["task"],
        T_in=10,
        T_out=10,
        model=model["name"],
        n_hidden=model["n_hidden"],
        n_layers=model["n_layers"],
        n_heads=model["n_heads"],
        act=model["act"],
        mlp_ratio=model["mlp_ratio"],
        dropout=model["dropout"],
        unified_pos=model["unified_pos"],
        ref=model["ref"],
        slice_num=model["slice_num"],
        modes=12,
        psi_dim=8,
        attn_type="nystrom",
        mwt_k=3,
        branch_depth=5,
        trunk_depth=6,
        hidden_channels=[],
        kernel_size=5,
        emb_dims=128,
        eval=0,
        save_name=train["save_name"],
        vis_num=cfg["inference"]["vis_num"],
        vis_bound=None,
    )


def get_device(args, dist):
    if dist.world_size == 1 and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    if dist.world_size > 1 and torch.cuda.is_available():
        return dist.device
    return torch.device("cpu")


def make_optimizer(args, model):
    if args.optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    if args.optimizer == "Adam":
        return torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    raise ValueError("Optimizer only AdamW or Adam")


def make_scheduler(args, optimizer, train_loader):
    if args.scheduler == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=args.pct_start,
        )
    if args.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    raise ValueError("Scheduler only OneCycleLR, CosineAnnealingLR or StepLR")


def evaluate(model, test_loader, dataset, args, device):
    myloss = L2Loss(size_average=False)
    model.eval()
    rel_err = 0.0
    with torch.no_grad():
        for pos, fx, y in test_loader:
            x, fx, y = pos.to(device), fx.to(device), y.to(device)
            if args.fun_dim == 0:
                fx = None
            out = model(x, fx)
            if args.normalize:
                out = dataset.y_normalizer.decode(out)
            rel_err += myloss(out, y).item()
    return rel_err / args.ntest


def train(args):
    load_onescience()
    DistributedManager.initialize()
    dist = DistributedManager()
    dataset, train_loader, test_loader, args.shapelist = get_data(args, dist)
    device = get_device(args, dist)
    model = get_model(args, device).to(device)
    if hasattr(dataset, "x_normalizer"):
        dataset.x_normalizer = dataset.x_normalizer.to(device)
    if hasattr(dataset, "y_normalizer"):
        dataset.y_normalizer = dataset.y_normalizer.to(device)

    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer, train_loader)
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.save_name}.pt"
    checkpoint_layers = (
        [layer.strip() for layer in args.checkpoint_layers.split(",") if layer.strip()]
        if args.use_checkpoint and args.checkpoint_layers
        else []
    )
    myloss = L2Loss(size_average=False)
    regloss = DerivLoss(size_average=False, shapelist=args.shapelist) if args.derivloss else None
    best_test_loss = float("inf")
    best_epoch = 0

    print(args)
    print(model)
    print(f"Use device: {device}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for pos, fx, y in train_loader:
            x, fx, y = pos.to(device), fx.to(device), y.to(device)
            if args.fun_dim == 0:
                fx = None
            with replace_function(
                module=model,
                replace_layers_list=checkpoint_layers,
                ddp_flag=(dist.world_size > 1),
            ):
                out = model(x, fx)
            if args.normalize:
                out = dataset.y_normalizer.decode(out)
                y = dataset.y_normalizer.decode(y)
            loss = myloss(out, y)
            if regloss is not None:
                loss = loss + 0.1 * regloss(out, y)

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if args.scheduler == "OneCycleLR":
                scheduler.step()

        if args.scheduler in {"CosineAnnealingLR", "StepLR"}:
            scheduler.step()

        train_loss = train_loss / args.ntrain
        rel_err = evaluate(model, test_loader, dataset, args, device)
        if rel_err < best_test_loss:
            best_test_loss = rel_err
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_test_loss": best_test_loss,
                    "best_epoch": best_epoch,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        if epoch % 10 == 0:
            print("Epoch {} Train loss : {:.5f}".format(epoch, train_loss))
            print("rel_err:{}".format(rel_err))

    print(
        "Training completed. Best model saved at epoch {} with rel_err: {:.5f}".format(
            best_epoch, best_test_loss
        )
    )


def main():
    os.chdir(PROJECT_ROOT)
    cfg = load_config()
    if cfg.get("runtime", {}).get("device", "cpu") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    train(build_args(cfg))


if __name__ == "__main__":
    main()
