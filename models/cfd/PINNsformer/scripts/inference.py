from __future__ import annotations

import argparse

import numpy as np
import torch

from onescience.utils.pinnsformer_util import get_data, make_time_sequence

from common import (
    build_model,
    exact_reaction_solution,
    load_config,
    project_path,
    relative_errors,
    select_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PINNsformer inference for the 1D reaction equation.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained checkpoint.")
    parser.add_argument("--device", default=None, help="Override runtime.device.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(args.device or cfg["runtime"]["device"])
    checkpoint = project_path(args.checkpoint or cfg["training"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run scripts/train.py first.")

    model = build_model(cfg).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    data_cfg = cfg["data"]
    res, _, _, _, _ = get_data(
        data_cfg["x_range"],
        data_cfg["t_range"],
        int(data_cfg["x_num"]),
        int(data_cfg["t_num"]),
    )
    seq = make_time_sequence(
        res,
        num_step=int(data_cfg["sequence"]["num_step"]),
        step=float(data_cfg["sequence"]["step"]),
    )
    res_tensor = torch.tensor(seq, dtype=torch.float32, requires_grad=False, device=device)
    x_test, t_test = res_tensor[:, :, 0:1], res_tensor[:, :, 1:2]

    with torch.no_grad():
        pred = model(x_test, t_test)[:, 0:1].detach().cpu().numpy().reshape(
            int(data_cfg["t_num"]),
            int(data_cfg["x_num"]),
        )

    x = res[:, 0].reshape(int(data_cfg["t_num"]), int(data_cfg["x_num"]))
    t = res[:, 1].reshape(int(data_cfg["t_num"]), int(data_cfg["x_num"]))
    target = exact_reaction_solution(x, t, cfg).astype(np.float32)
    metrics = relative_errors(pred, target)

    output_path = project_path(cfg["paths"]["prediction"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        x=x.astype(np.float32),
        t=t.astype(np.float32),
        prediction=pred.astype(np.float32),
        target=target,
        absolute_error=np.abs(pred - target).astype(np.float32),
        relative_l1=metrics["relative_l1"],
        relative_l2=metrics["relative_l2"],
    )
    print(f"prediction saved to {output_path}")
    print(f"relative L1 error: {metrics['relative_l1']:.6f}")
    print(f"relative L2 error: {metrics['relative_l2']:.6f}")


if __name__ == "__main__":
    main()
