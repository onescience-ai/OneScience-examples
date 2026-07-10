import argparse
import json
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import build_model, infer_task_type
import onescience
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="Train CFDBench static or autoregressive models.")
    parser.add_argument("--model", default=None, help="Override root.model.name, e.g. ffn, deeponet, fno, auto_ffn.")
    return parser.parse_args()


def load_config(model_name=None):
    cfg = YParams(str(PROJECT_ROOT / "conf" / "config.yaml"), "root")
    if model_name:
        cfg.model.name = model_name
    elif os.environ.get("CFDBENCH_MODEL_NAME"):
        cfg.model.name = os.environ["CFDBENCH_MODEL_NAME"]
    cfg.datapipe.source.data_dir = str(resolve_path(cfg.datapipe.source.data_dir))
    cfg.training.output_dir = str(resolve_path(cfg.training.output_dir))
    return cfg


def checkpoint_path(cfg):
    name = cfg.training.get("checkpoint_name", "auto")
    if name == "auto":
        name = f"{cfg.model.name}.pt"
    return PROJECT_ROOT / "weight" / name


def output_path(cfg, task_type):
    output_dir = Path(cfg.training.output_dir)
    if cfg.training.get("group_by_model", False):
        output_dir = output_dir / task_type / cfg.model.name
    return output_dir


def select_device(requested, dist):
    if requested == "auto":
        return dist.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is not available.")
    return torch.device(requested)


def dump_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_cfdbench_datapipe_class():
    runtime_root = Path(onescience.__file__).resolve().parent
    datapipe_file = runtime_root / "datapipes" / "cfd" / "cfdbench.py"
    spec = importlib.util.spec_from_file_location("_onescience_cfdbench_datapipe", datapipe_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CFDBench datapipe from {datapipe_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CFDBenchDatapipe


def mean_scores(score_lists):
    return {key: float(np.mean(values)) for key, values in score_lists.items() if values}


def evaluate(model, loader, device, dist, desc="Evaluating"):
    model_eval = model.module if hasattr(model, "module") else model
    model_eval.eval()
    score_lists = {name: [] for name in model_eval.loss_fn.get_score_names()}

    with torch.no_grad():
        iterator = tqdm(loader, desc=desc, disable=(dist.rank != 0))
        for batch in iterator:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model_eval(**batch)
            for key, value in outputs["loss"].items():
                score_lists[key].append(float(value.detach().cpu()))

    return {"mean": mean_scores(score_lists), "all": score_lists}


def train(cfg, model, datapipe, output_dir, device, dist):
    train_cfg = cfg.training
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, _ = datapipe.val_dataloader()

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Increase fake_data.num_cases_per_subset or reduce batch_size.")

    optimizer = Adam(model.parameters(), lr=train_cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=train_cfg.lr_step_size, gamma=train_cfg.lr_gamma)
    train_losses = []
    best_nmse = float("inf")

    for epoch in range(train_cfg.num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        start = time.time()

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(dist.rank != 0))
        for step, batch in enumerate(iterator):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"][train_cfg.loss_name]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            train_losses.append(loss_value)
            if dist.rank == 0 and (step + 1) % train_cfg.log_interval == 0:
                iterator.set_postfix({"loss": f"{loss_value:.4e}"})

        scheduler.step()

        if dist.rank == 0 and (epoch + 1) % train_cfg.eval_interval == 0:
            ckpt_dir = output_dir / f"ckpt-{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            scores = evaluate(model, val_loader, device, dist, desc=f"Val {epoch}")
            dump_json(scores, ckpt_dir / "dev_scores.json")
            dump_json({"epoch": epoch, "train_loss": train_losses, "seconds": time.time() - start}, ckpt_dir / "scores.json")

            nmse = scores["mean"].get(train_cfg.loss_name, float("inf"))
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), ckpt_dir / "model.pt")
            if nmse <= best_nmse:
                best_nmse = nmse
                weight_path = checkpoint_path(cfg)
                torch.save(model_to_save.state_dict(), weight_path)
                print(f"Saved best checkpoint to {weight_path}")

        if dist.world_size > 1:
            torch.distributed.barrier()

    if dist.rank == 0:
        dump_json(train_losses, output_dir / "train_losses.json")


def main(required_task_type="static", entry_name="scripts/train.py"):
    args = parse_args()
    DistributedManager.initialize()
    dist = DistributedManager()
    cfg = load_config(args.model)
    task_type = infer_task_type(cfg.model.name)
    if required_task_type is not None and task_type != required_task_type:
        expected = "ffn/deeponet" if required_task_type == "static" else "auto_* / resnet / unet / fno"
        raise ValueError(
            f"{entry_name} is the {required_task_type} entry, but model.name={cfg.model.name!r} is {task_type}. "
            f"Use a {expected} model, or run scripts/train_auto.py for autoregressive models."
        )
    cfg.datapipe.data.task_type = task_type
    device = select_device(cfg.training.get("device", "auto"), dist)

    output_dir = output_path(cfg, task_type)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Config: {PROJECT_ROOT / 'conf' / 'config.yaml'}")
        print(f"Model: {cfg.model.name} ({task_type})")
        print(f"Data: {cfg.datapipe.source.data_dir}")
        print(f"Output: {output_dir}")
        print(f"Checkpoint: {checkpoint_path(cfg)}")

    torch.set_num_threads(1)
    CFDBenchDatapipe = load_cfdbench_datapipe_class()
    datapipe = CFDBenchDatapipe(cfg.datapipe, distributed=(dist.world_size > 1))
    model = build_model(cfg).to(device)

    if dist.world_size > 1 and "train" in cfg.training.mode:
        device_ids = [dist.local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids)

    if "train" in cfg.training.mode:
        train(cfg, model, datapipe, output_dir, device, dist)

    if "test" in cfg.training.mode and dist.rank == 0:
        scores = evaluate(model, datapipe.test_dataloader(), device, dist, desc="Test")
        dump_json(scores, output_dir / "test_scores.json")
        print(f"Test scores: {scores['mean']}")

    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
