import argparse
import json
import importlib.util
import os
import sys
from pathlib import Path

import torch
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
    parser = argparse.ArgumentParser(description="Run CFDBench inference.")
    parser.add_argument("--model", default=None, help="Override root.model.name and read weight/<model>.pt by default.")
    return parser.parse_args()


def load_config(model_name=None):
    cfg = YParams(str(PROJECT_ROOT / "conf" / "config.yaml"), "root")
    if model_name:
        cfg.model.name = model_name
    elif os.environ.get("CFDBENCH_MODEL_NAME"):
        cfg.model.name = os.environ["CFDBENCH_MODEL_NAME"]
    cfg.datapipe.source.data_dir = str(resolve_path(cfg.datapipe.source.data_dir))
    cfg.inference.output_dir = str(resolve_path(cfg.inference.output_dir))
    return cfg


def checkpoint_path(cfg):
    path_value = cfg.inference.get("checkpoint_path", "auto")
    if path_value == "auto":
        return PROJECT_ROOT / "weight" / f"{cfg.model.name}.pt"
    return resolve_path(path_value)


def output_path(cfg, task_type):
    output_dir = Path(cfg.inference.output_dir)
    if cfg.inference.get("group_by_model", False):
        output_dir = output_dir / task_type / cfg.model.name
    return output_dir


def select_device(requested, dist):
    if requested == "auto":
        return dist.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is not available.")
    return torch.device(requested)


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


def predict_full_frame(model, batch, task_type):
    if task_type == "auto":
        return model.generate(
            inputs=batch["inputs"],
            case_params=batch["case_params"],
            mask=batch.get("mask"),
        )

    labels = batch["label"]
    return model.generate_one(
        case_params=batch["case_params"],
        t=batch["t"],
        height=labels.shape[-2],
        width=labels.shape[-1],
    )


def main():
    args = parse_args()
    DistributedManager.initialize()
    dist = DistributedManager()
    cfg = load_config(args.model)
    task_type = infer_task_type(cfg.model.name)
    cfg.datapipe.data.task_type = task_type
    device = select_device(cfg.inference.get("device", "auto"), dist)

    ckpt_path = checkpoint_path(cfg)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Run scripts/train.py first or update root.inference.checkpoint_path.")

    torch.set_num_threads(1)
    CFDBenchDatapipe = load_cfdbench_datapipe_class()
    datapipe = CFDBenchDatapipe(cfg.datapipe, distributed=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    output_dir = output_path(cfg, task_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = []
    score_lists = {name: [] for name in model.loss_fn.get_score_names()}
    with torch.no_grad():
        for step, batch in enumerate(tqdm(datapipe.test_dataloader(), desc="Inferring")):
            if step >= cfg.inference.max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = predict_full_frame(model, batch, task_type)
            label = batch["label"][:, : pred.shape[1]]
            loss = model.loss_fn(preds=pred, labels=label)
            preds.append(pred.detach().cpu())
            for key, value in loss.items():
                score_lists[key].append(float(value.detach().cpu()))

    if not preds:
        raise RuntimeError("Inference loader is empty. Check dataset split and data directory.")

    torch.save(torch.cat(preds, dim=0), output_dir / "preds.pt")
    means = {key: sum(values) / len(values) for key, values in score_lists.items() if values}
    with open(output_dir / "scores.json", "w", encoding="utf-8") as f:
        json.dump({"mean": means, "all": score_lists}, f, indent=2)
    print(f"Saved predictions to {output_dir / 'preds.pt'}")
    print(f"Scores: {means}")

    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
