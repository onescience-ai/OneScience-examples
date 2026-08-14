#!/usr/bin/env bash
set -euo pipefail

demo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eqv3_dir="$(cd "$demo_dir/.." && pwd)"
parser="$demo_dir/_parse_config.py"
config=""
submit=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="$2"; shift 2 ;;
    --config=*) config="${1#*=}"; shift ;;
    --submit) submit=true; shift ;;
    -h|--help) echo "Usage: bash demo/run.sh --config configs/<name>.yaml"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$config" ]] || { echo "Please specify --config configs/<name>.yaml" >&2; exit 2; }
[[ "$config" = /* ]] || config="$demo_dir/$config"
[[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 2; }
[[ -n "${CONDA_PREFIX:-}" ]] || { echo "Activate a OneScience MatChem conda environment first." >&2; exit 2; }

export ONESCIENCE_EQUIFORMER_V3_DIR="$eqv3_dir"
export ONESCIENCE_EQUIFORMER_V3_JD_PATH="${ONESCIENCE_EQUIFORMER_V3_JD_PATH:-${ONESCIENCE_MODELS_DIR:?}/EquiformerV3/Jd.pt}"
export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-$(basename "$CONDA_PREFIX")}"

name="$(python3 "$parser" "$config" name)"
eval "$(python3 "$parser" "$config" launch)"
eval "$(python3 "$parser" "$config" slurm)"
env_exports="$(python3 "$parser" "$config" env)"
[[ "$RUN_MODE" == submit ]] && submit=true

timestamp="$(date +%Y%m%d_%H%M%S)"
output_root="${EQUIFORMER_V3_OUTPUT_ROOT:-$eqv3_dir/outputs}"
output_dir="$output_root/${name}_${timestamp}"
mkdir -p "$output_dir/checkpoints"
cp "$config" "$output_dir/config.yaml"
python3 "$parser" "$config" finetune-config > "$output_dir/finetune.yaml"

eval "$env_exports"
cd "$output_dir"
if [[ "$submit" == true ]]; then
  echo "Submitting is supported by the generated YAML launch settings; run from a configured Slurm allocation."
  exit 2
fi
if (( GPUS_PER_NODE > 1 )); then
  exec torchrun --standalone --nproc_per_node="$GPUS_PER_NODE" "$eqv3_dir/finetune.py" --config "$output_dir/finetune.yaml"
fi
exec python "$eqv3_dir/finetune.py" --config "$output_dir/finetune.yaml"
