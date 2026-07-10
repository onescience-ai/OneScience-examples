import json
import os
from pathlib import Path

import train as train_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_model(args, device):
    model = train_runtime.get_model(args, device).to(device)
    checkpoint_path = PROJECT_ROOT / "checkpoints" / f"{args.save_name}.pt"
    state = train_runtime.torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    model.load_state_dict(model_state)
    model.eval()
    return model


def evaluate(model, test_loader, dataset, args, device):
    loss_func = train_runtime.L2Loss(size_average=False)
    totals = {
        "rel_err": 0.0,
        "abs_err": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "maxae": 0.0,
        "r2": 0.0,
    }
    count = 0
    with train_runtime.torch.no_grad():
        for pos, fx, y in test_loader:
            x, fx, y = pos.to(device), fx.to(device), y.to(device)
            if args.fun_dim == 0:
                fx = None
            out = model(x, fx)
            if args.normalize:
                out = dataset.y_normalizer.decode(out)

            batch_size = y.shape[0]
            totals["rel_err"] += loss_func.rel(out, y).item()
            totals["abs_err"] += loss_func.abs(out, y).item()
            totals["mse"] += loss_func.MSE(out, y).item()
            totals["mae"] += loss_func.MAE(out, y).item()
            totals["maxae"] += loss_func.MaxAE(out, y).item()
            totals["r2"] += loss_func.R2Score(out, y).item()
            count += batch_size

    return {name: value / args.ntest for name, value in totals.items()}


def main():
    os.chdir(PROJECT_ROOT)
    cfg = train_runtime.load_config()
    if cfg.get("runtime", {}).get("device", "cpu") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    weight_path = PROJECT_ROOT / cfg["paths"]["weight_path"]
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weight_path}. Run python scripts/train.py first "
            "or update paths.weight_path/save_name in config/config.yaml."
        )

    args = train_runtime.build_args(cfg)
    args.eval = cfg["inference"]["eval"]
    train_runtime.load_onescience()
    train_runtime.DistributedManager.initialize()
    dist = train_runtime.DistributedManager()
    dataset, _, test_loader, args.shapelist = train_runtime.get_data(args, dist)
    device = train_runtime.get_device(args, dist)
    if hasattr(dataset, "x_normalizer"):
        dataset.x_normalizer = dataset.x_normalizer.to(device)
    if hasattr(dataset, "y_normalizer"):
        dataset.y_normalizer = dataset.y_normalizer.to(device)

    model = load_model(args, device)
    metrics = evaluate(model, test_loader, dataset, args, device)

    result_dir = PROJECT_ROOT / cfg["paths"]["result_dir"] / args.save_name
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n===== 测试结果 =====")
    print(f"平均相对误差: {metrics['rel_err']:.6e}")
    print(f"平均绝对误差: {metrics['abs_err']:.6e}")
    print(f"平均MSE: {metrics['mse']:.6e}")
    print(f"平均MAE: {metrics['mae']:.6e}")
    print(f"平均MaxAE: {metrics['maxae']:.6e}")
    print(f"平均R2分数: {metrics['r2']:.6f}")
    print("====================")


if __name__ == "__main__":
    main()
