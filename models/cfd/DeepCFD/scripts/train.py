import sys
import importlib.util
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import build_model
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams
import onescience


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config():
    cfg = YParams(str(PROJECT_ROOT / "config" / "config.yaml"), "root")
    cfg.datapipe.source.data_dir = str(resolve_path(cfg.datapipe.source.data_dir))
    cfg.training.output_dir = str(resolve_path(cfg.training.output_dir))
    return cfg


def load_deepcfd_datapipe_class():
    runtime_root = Path(onescience.__file__).resolve().parent
    datapipe_file = runtime_root / "datapipes" / "cfd" / "deepcfd.py"
    spec = importlib.util.spec_from_file_location("_onescience_deepcfd_datapipe", datapipe_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DeepCFD datapipe from {datapipe_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DeepCFDDatapipe


def loss_func(output, target, weights):
    lossu = (output[:, 0] - target[:, 0]) ** 2
    lossv = (output[:, 1] - target[:, 1]) ** 2
    lossp = torch.abs(output[:, 2] - target[:, 2])
    loss_stack = torch.stack([lossu, lossv, lossp], dim=1)
    return torch.sum(loss_stack / weights)


def evaluate(model, loader, device, weights, dist):
    model.eval()
    total_loss = 0.0
    total_ux_mse = 0.0
    total_uy_mse = 0.0
    total_p_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        iterator = tqdm(loader, desc="Evaluating", disable=(dist.rank != 0))
        for batch in iterator:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            output = model(x)

            total_loss += loss_func(output, y, weights).item()
            total_ux_mse += torch.sum((output[:, 0] - y[:, 0]) ** 2).item()
            total_uy_mse += torch.sum((output[:, 1] - y[:, 1]) ** 2).item()
            total_p_mse += torch.sum((output[:, 2] - y[:, 2]) ** 2).item()
            num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Evaluation loader is empty. Check split_ratio and dataset size.")
    return total_loss / num_batches, total_ux_mse, total_uy_mse, total_p_mse


def main():
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    cfg = load_config()
    DeepCFDDatapipe = load_deepcfd_datapipe_class()

    output_dir = Path(cfg.training.output_dir)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Config: {PROJECT_ROOT / 'config' / 'config.yaml'}")
        print(f"Data: {cfg.datapipe.source.data_dir}")
        print(f"Checkpoint directory: {output_dir}")

    datapipe = DeepCFDDatapipe(cfg.datapipe, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    test_loader, _ = datapipe.test_dataloader()
    loss_weights = datapipe.get_loss_weights().to(device)

    model = build_model(cfg.model).to(device)
    if dist.world_size > 1:
        device_ids = [dist.local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.training.num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(dist.rank != 0))

        for batch in iterator:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = loss_func(output, y, loss_weights)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if dist.rank == 0:
                iterator.set_postfix({"loss": f"{loss.item():.4e}"})

        if len(train_loader) == 0:
            raise RuntimeError("Training loader is empty. Check split_ratio and dataset size.")
        avg_train_loss = train_loss / len(train_loader)

        if (epoch + 1) % cfg.training.eval_interval == 0:
            val_loss, ux_err, uy_err, p_err = evaluate(model, test_loader, device, loss_weights, dist)

            if dist.rank == 0:
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4e} | Val Loss: {val_loss:.4e}")
                print(f"Metrics (Sum Sq Err): Ux={ux_err:.2e}, Uy={uy_err:.2e}, P={p_err:.2e}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    model_to_save = model.module if hasattr(model, "module") else model
                    ckpt = {
                        "model_state": model_to_save.state_dict(),
                        "config": cfg.model.to_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    }
                    torch.save(ckpt, output_dir / cfg.training.checkpoint_name)
                    print(f"Saved best model to {output_dir / cfg.training.checkpoint_name}")
                else:
                    patience_counter += 1

            stop_flag = torch.tensor([0], device=device)
            if dist.rank == 0 and patience_counter >= cfg.training.patience:
                stop_flag += 1
            if dist.world_size > 1:
                torch.distributed.broadcast(stop_flag, src=0)
            if stop_flag.item() > 0:
                break

    dist.cleanup()


if __name__ == "__main__":
    main()
