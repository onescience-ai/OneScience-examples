from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_laplace_data,
    checkpoint_state,
    load_config,
    project_path,
    relative_l2,
    resolve_device,
    resolve_dtype,
    seed_everything,
)
from model.bpinn import (  # noqa: E402
    build_model,
    laplace1d_loss_components,
    weighted_loss,
)


DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


def main() -> None:
    config_path = DEFAULT_CONFIG.resolve()
    config = load_config(config_path)
    common = config["common"]
    training = config["training"]
    lbfgs_iters = int(training["lbfgs_iters"])
    refinement_lr = float(training["refinement_lr"])
    if lbfgs_iters <= 0 or refinement_lr <= 0:
        raise ValueError("lbfgs-iters and lr must be positive")
    device = resolve_device(str(common["device"]))
    dtype = resolve_dtype(str(common["dtype"]))
    weight_dir = project_path(common["weight_dir"], PROJECT_ROOT)
    input_path = weight_dir / training["checkpoint_name"]
    output_path = weight_dir / training["refined_checkpoint_name"]
    if not input_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    state, metadata = checkpoint_state(checkpoint)
    model_config = metadata.get("model_config", config["model"])
    data_config = metadata.get("data_config", config["data"])
    loss_weights = metadata.get("loss_weights", config["loss"])
    seed = int(metadata.get("seed", common["seed"]))
    seed_everything(seed)
    data = build_laplace_data(data_config, seed, device, dtype)
    model = build_model(model_config, dtype=dtype).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True)

    def loss_value() -> torch.Tensor:
        components = laplace1d_loss_components(
            model,
            data["x_solution"],
            data["u_solution"],
            data["x_boundary"],
            data["u_boundary"],
            data["x_pde"],
        )
        return weighted_loss(components, loss_weights)

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=refinement_lr,
        max_iter=lbfgs_iters,
        max_eval=max(1, 2 * lbfgs_iters),
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        loss = loss_value()
        if not torch.isfinite(loss):
            raise FloatingPointError("BPINN refinement loss became non-finite")
        loss.backward()
        return loss

    print(f"Input checkpoint: {input_path}")
    print(f"Device: {device}")
    lbfgs.step(closure)
    with torch.no_grad():
        prediction = model.predict_u(data["x_test"])
    error = relative_l2(prediction, data["u_test"])
    refined = dict(metadata)
    refined.update(
        {
            "case": "laplace1d",
            "architecture": "bpinn",
            "model_state": model.state_dict(),
            "model_config": model_config,
            "data_config": data_config,
            "loss_weights": loss_weights,
            "seed": seed,
            "refinement_iters": lbfgs_iters,
            "final_l2": error,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refined, output_path)
    print(f"Relative L2 after refinement={error:.6e}")
    print(f"Saved checkpoint: {output_path}")


if __name__ == "__main__":
    main()
