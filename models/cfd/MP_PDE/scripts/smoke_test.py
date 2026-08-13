"""Bounded integration smoke test; creates artifacts only in a temporary directory."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dataset import E3Dataset, generate_e3_hdf5  # noqa: E402
from models.pde import periodic_neighbor_indices  # noqa: E402
from scripts.inference import atomic_json, atomic_npz, compute_metrics  # noqa: E402
from scripts.train import build_model, load_config, rmse_loss, rollout_batch  # noqa: E402


def main() -> None:
    canonical = load_config(PROJECT_ROOT / "config/config.yaml")
    assert canonical["model"]["hidden_dim"] == 164
    assert canonical["model"]["time_window"] == 25
    neighbors = periodic_neighbor_indices(40, canonical["model"]["neighbor_offsets"])
    assert neighbors.shape == (40, 6)
    assert torch.all(torch.tensor([row.unique().numel() == 6 for row in neighbors]))

    with tempfile.TemporaryDirectory(prefix="mp_pde_smoke_") as temporary_directory:
        root = Path(temporary_directory)
        config = copy.deepcopy(canonical)
        config["data"]["num_time_points"] = 50
        config["data"]["high_resolution_nx"] = 40
        config["data"]["resolution"] = 40
        config["data"]["train_samples"] = 1
        config["data"]["valid_samples"] = 1
        config["data"]["test_samples"] = 1
        config["data"]["parallel_generation"].update({"workers": 2, "max_in_flight": 2, "flush_every": 1})
        config["visualization"]["time_indices"] = [25, 30, 40, 49]
        data_path = root / "e3_smoke.h5"
        generate_e3_hdf5(config, data_path, sample_counts={"train": 1, "valid": 1, "test": 1})
        partial_path = data_path.with_suffix(data_path.suffix + ".partial")
        data_path.replace(partial_path)
        generate_e3_hdf5(config, data_path, sample_counts={"train": 1, "valid": 1, "test": 1})
        dataset = E3Dataset(data_path, "test", expected_nt=50, expected_nx=40)
        batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(config).to(device)
        window = 25
        history = batch["u"][:, :window].transpose(1, 2).to(device)
        target = batch["u"][:, window : 2 * window].transpose(1, 2).to(device)
        times = batch["t"]
        prediction = model(
            history, batch["x"].to(device), times[:, window - 1].to(device), batch["params"].to(device),
            (times[:, 1] - times[:, 0]).to(device),
        )
        assert prediction.shape == (1, 40, 25)
        loss = rmse_loss(prediction, target, 1.0e-12)
        loss.backward()
        assert np.isfinite(float(loss.detach().cpu()))
        checkpoint_path = root / "smoke_checkpoint.pth"
        torch.save({"model_state": model.state_dict(), "resolved_config": config}, checkpoint_path)
        reloaded = build_model(config).to(device)
        reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state"], strict=True)
        reloaded.eval()
        rollout = rollout_batch(reloaded, batch, device, window).cpu().numpy()
        target_array = batch["u"].numpy()
        metrics = compute_metrics(rollout, target_array, window)
        assert np.isfinite(metrics["accumulated_mse"])

        results = root / "results"
        per_time_mse = metrics.pop("per_time_mse")
        atomic_npz(
            results / "predictions.npz", prediction=rollout, target=target_array, x=batch["x"][0].numpy(),
            t=batch["t"][0].numpy(), params=batch["params"].numpy(), sample_indices=batch["index"].numpy(),
            forecast_start_index=np.asarray(window, dtype=np.int64), per_time_mse=per_time_mse,
        )
        atomic_json(results / "metrics.json", {**metrics, "samples": 1})
        with (results / "train_history.json").open("w", encoding="utf-8") as stream:
            json.dump([{"epoch": 0, "train_rmse": float(loss.detach().cpu()), "validation_bundle_rmse": float(loss.detach().cpu()), "validation_accumulated_mse": metrics["accumulated_mse"]}], stream)
        config["paths"].update(
            {"predictions": str(results / "predictions.npz"), "metrics": str(results / "metrics.json"), "train_history": str(results / "train_history.json"), "results": str(results)}
        )
        smoke_config = root / "config.yaml"
        with smoke_config.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/result.py"), "--config", str(smoke_config)], check=True)
        for name in ("e3_rollout.png", "e3_error.png", "training_curve.png"):
            assert (results / name).is_file() and (results / name).stat().st_size > 0
        print(
            f"smoke_ok device={device} loss={float(loss.detach().cpu()):.8e} "
            f"accumulated_mse={metrics['accumulated_mse']:.8e}", flush=True,
        )


if __name__ == "__main__":
    main()
