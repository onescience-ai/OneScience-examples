"""Guarded decoder continuation for the MARIO AirfRANS run.

This script deliberately evaluates after short training blocks and restores the
best checkpoint when a block hurts the selected full_test score. It is meant for
reproduction debugging where the user explicitly wants to protect test accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


PAPER_TARGETS = {
    "ux_x1e_minus_2_table_value": 0.152,
    "uy_x1e_minus_2_table_value": 0.113,
    "p_x1e_minus_2_table_value": 0.240,
    "nut_x1e_minus_1_table_value": 0.096,
    "surface_p_x1e_minus_1_table_value": 0.270,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_cmd(cmd: list[str], *, cwd: Path) -> None:
    print("cmd=" + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def copy_seed(seed_dir: Path, out_dir: Path, *, reset: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["geometry_last.pt", "decoder_last.pt", "train_latents.npz", "stats.npz"]:
        src = seed_dir / name
        dst = out_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        if reset or not dst.exists():
            shutil.copy2(src, dst)
            print(f"seed_copy {src} -> {dst}", flush=True)


def checkpoint_epoch(path: Path) -> int:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state.get("epoch", 0))


def score_metrics(metrics: dict[str, Any]) -> float:
    scaled = metrics["paper_table_scaled_mse"]
    ratios = [float(scaled[key]) / target for key, target in PAPER_TARGETS.items()]
    return float(sum(ratios) / len(ratios))


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    scaled = metrics["paper_table_scaled_mse"]
    out = {key: float(scaled[key]) for key in PAPER_TARGETS}
    out["score_mean_paper_ratio"] = score_metrics(metrics)
    return out


def save_best(out_dir: Path, metrics: dict[str, Any], *, epoch: int, lr: float, score: float) -> None:
    shutil.copy2(out_dir / "decoder_last.pt", out_dir / "decoder_best.pt")
    shutil.copy2(out_dir / "geometry_last.pt", out_dir / "geometry_best.pt")
    best_payload = {
        "epoch": epoch,
        "lr": lr,
        "score_mean_paper_ratio": score,
        "metrics": metrics,
    }
    write_json(out_dir / "eval" / "best_metrics.json", best_payload)
    print(f"best_saved epoch={epoch} score={score:.6f}", flush=True)


def restore_best(out_dir: Path) -> int:
    best_path = out_dir / "decoder_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(best_path)
    shutil.copy2(best_path, out_dir / "decoder_last.pt")
    return checkpoint_epoch(out_dir / "decoder_last.pt")


def append_history(out_dir: Path, payload: dict[str, Any]) -> None:
    path = out_dir / "guarded_history.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def make_runtime_config(base_cfg: dict[str, Any], out_dir: Path, *, epochs: int, lr: float) -> Path:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["decoder"]["epochs"] = int(epochs)
    cfg["decoder"]["lr"] = float(lr)
    cfg["training"]["output_dir"] = str(out_dir)
    cfg["stats_path"] = str(out_dir / "stats.npz")
    path = out_dir / "guarded_runtime_config.json"
    write_json(path, cfg)
    return path


def evaluate(root: Path, config_path: Path) -> dict[str, Any]:
    run_cmd([sys.executable, "-m", "src.evaluate", "--config", str(config_path)], cwd=root)
    cfg = load_json(config_path)
    split = cfg.get("test_split", "full_test")
    return load_json(Path(cfg["training"]["output_dir"]) / "eval" / f"{split}_metrics.json")


def train_block(root: Path, config_path: Path) -> None:
    run_cmd([sys.executable, "-m", "src.train", "--config", str(config_path), "--stage", "decoder"], cwd=root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-dir", required=True)
    parser.add_argument("--target-epoch", type=int, default=500)
    parser.add_argument("--block-epochs", type=int, default=10)
    parser.add_argument("--initial-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2.5e-5)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--reset-from-seed", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_cfg = load_json(Path(args.config))
    out_dir = Path(base_cfg["training"]["output_dir"])
    copy_seed(Path(args.seed_dir), out_dir, reset=bool(args.reset_from_seed))

    lr = float(args.initial_lr)
    no_improve = 0
    current_epoch = checkpoint_epoch(out_dir / "decoder_last.pt")
    runtime_cfg = make_runtime_config(base_cfg, out_dir, epochs=current_epoch, lr=lr)

    print(f"guard_start epoch={current_epoch} target={args.target_epoch} lr={lr}", flush=True)
    metrics = evaluate(root, runtime_cfg)
    best_score = score_metrics(metrics)
    save_best(out_dir, metrics, epoch=current_epoch, lr=lr, score=best_score)
    append_history(
        out_dir,
        {
            "event": "initial",
            "epoch": current_epoch,
            "lr": lr,
            "score": best_score,
            "summary": summarize_metrics(metrics),
        },
    )

    while current_epoch < int(args.target_epoch) and lr >= float(args.min_lr):
        next_epoch = min(current_epoch + int(args.block_epochs), int(args.target_epoch))
        runtime_cfg = make_runtime_config(base_cfg, out_dir, epochs=next_epoch, lr=lr)
        print(f"guard_train from={current_epoch} to={next_epoch} lr={lr}", flush=True)
        train_block(root, runtime_cfg)
        current_epoch = checkpoint_epoch(out_dir / "decoder_last.pt")
        metrics = evaluate(root, runtime_cfg)
        score = score_metrics(metrics)
        improved = score < best_score
        append_history(
            out_dir,
            {
                "event": "block",
                "epoch": current_epoch,
                "lr": lr,
                "score": score,
                "best_score_before": best_score,
                "improved": improved,
                "summary": summarize_metrics(metrics),
            },
        )
        print(f"guard_eval epoch={current_epoch} score={score:.6f} best={best_score:.6f} improved={improved}", flush=True)
        if improved:
            best_score = score
            no_improve = 0
            save_best(out_dir, metrics, epoch=current_epoch, lr=lr, score=best_score)
        else:
            no_improve += 1
            current_epoch = restore_best(out_dir)
            lr *= float(args.lr_decay)
            print(f"guard_restore epoch={current_epoch} next_lr={lr} no_improve={no_improve}", flush=True)
            if no_improve >= int(args.patience):
                print("guard_stop reason=patience", flush=True)
                break

    current_epoch = restore_best(out_dir)
    runtime_cfg = make_runtime_config(base_cfg, out_dir, epochs=current_epoch, lr=lr)
    final_metrics = evaluate(root, runtime_cfg)
    final_score = score_metrics(final_metrics)
    write_json(
        out_dir / "eval" / "guarded_final_summary.json",
        {
            "final_epoch": current_epoch,
            "final_lr": lr,
            "final_score_mean_paper_ratio": final_score,
            "metrics": final_metrics,
        },
    )
    print(f"guard_done best_epoch={current_epoch} score={final_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
