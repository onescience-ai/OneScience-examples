#!/usr/bin/env bash
set -eo pipefail

source ~/env.sh

ROOT="/public/home/liuyushuang/code/onecode_new_model/paper_2505_14704/mario"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CONFIG="${ROOT}/configs/airfrans_mario_e02r1n06_paper_aligned.json"
LOG="${ROOT}/logs/e02r1n06_paper_aligned_train_eval.log"
mkdir -p "${ROOT}/logs"

{
  echo "host=$(hostname)"
  echo "time=$(date '+%F %T %z')"
  python - <<'PY'
from pathlib import Path
import json
import os
import shutil
import torch

root = Path("/public/home/liuyushuang/code/onecode_new_model/paper_2505_14704/mario")
config = root / "configs" / "airfrans_mario_e02r1n06_paper_aligned.json"
cfg = json.loads(config.read_text(encoding="utf-8"))
out_dir = Path(cfg["training"]["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
reset = os.environ.get("RESET_FROM_SEED", "0") == "1"

seed_dirs = [
    root / "outputs" / "mario_e02r1n06_guarded",
    root / "outputs" / "mario_e02r1n06_fast",
    root / "outputs" / "mario_e02r1n06_100ep",
]
required = ["geometry_last.pt", "decoder_last.pt", "train_latents.npz", "stats.npz"]
if reset:
    for name in [*required, "decoder_best.pt", "decoder_validation_history.jsonl"]:
        path = out_dir / name
        if path.exists():
            path.unlink()
            print(f"reset_unlink {path}")
for name in required:
    dst = out_dir / name
    if dst.exists():
        continue
    for seed_dir in seed_dirs:
        src = seed_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"seed_copy {src} -> {dst}")
            break
    else:
        print(f"seed_missing {name}; training may need the earlier stage")

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"cuda_device_{idx}={torch.cuda.get_device_name(idx)}")
PY
  python -m src.train --config "${CONFIG}" --stage decoder
  python -m src.evaluate --config "${CONFIG}"
} 2>&1 | tee -a "${LOG}"
