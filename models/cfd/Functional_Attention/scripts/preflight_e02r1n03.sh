#!/usr/bin/env bash
set -eo pipefail

source "${HOME}/env.sh"
cd /public/home/liuyushuang/code/onecode_new_model/paper_2605_31559_original_airfrans

echo "host=$(hostname)"
echo "time=$(date '+%F %T %z')"
echo "python=$(command -v python)"

python - <<'PY'
from pathlib import Path
import torch

ckpt = Path("weight/funcattn_reynolds.pt")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"cuda_device0={torch.cuda.get_device_name(0)}")
if ckpt.exists():
    state = torch.load(ckpt, map_location="cpu")
    print(f"checkpoint_epoch={state.get('epoch')}")
    print(f"checkpoint_last_loss={state.get('last_loss')}")
PY

pgrep -af "paper_2605_31559_original_airfrans.*scripts/train.py" || true

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
elif command -v hy-smi >/dev/null 2>&1; then
    hy-smi
else
    echo "no_gpu_smi_command_found"
fi
