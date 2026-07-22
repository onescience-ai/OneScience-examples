#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${GRAPHCAST_SOURCE_DIR:-${SCRIPT_DIR}}"
DATA_ROOT="${WEATHER_DATA_ROOT:-/root/group_data/SDU-Test/weather}"
ASSETS_DIR="${WEATHER_ASSETS_DIR:-${DATA_ROOT}/assets}"
OUTPUT_DIR="${WEATHER_OUTPUT_DIR:-${DATA_ROOT}/outputs}"
LOGS_DIR="${WEATHER_LOGS_DIR:-${DATA_ROOT}/logs}"
VENV_DIR="${WEATHER_VENV:-${HOME}/.venvs/weather}"
MODEL="${WEATHER_MODEL:-GraphCast_small}"
DATASET="${WEATHER_DATASET:-}"
STEPS="${WEATHER_STEPS:-1}"

mkdir -p "${OUTPUT_DIR}" "${LOGS_DIR}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
elif [[ -f "${VENV_DIR}/Scripts/activate" ]]; then
  # Git Bash with a Windows virtual/Conda environment.
  # shellcheck disable=SC1091
  source "${VENV_DIR}/Scripts/activate"
elif ! command -v python >/dev/null 2>&1; then
  echo "ERROR: virtual environment not found at ${VENV_DIR}, and python is unavailable." >&2
  exit 2
else
  echo "WARNING: ${VENV_DIR} was not found; using python from PATH." >&2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
environment_log="${LOGS_DIR}/environment_${timestamp}.log"
echo "Checking repository, Python, JAX and devices first..."
PYTHONPATH="${SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${SCRIPT_DIR}/environment_check.py" \
    --source-dir "${SOURCE_DIR}" \
    --output "${environment_log}"

shopt -s nullglob
params=("${ASSETS_DIR}"/params/*GraphCast_small*.npz)
datasets=("${ASSETS_DIR}"/dataset/source-era5*_res-1.0_levels-13_steps-01.nc)
stats=(
  "${ASSETS_DIR}/stats/diffs_stddev_by_level.nc"
  "${ASSETS_DIR}/stats/mean_by_level.nc"
  "${ASSETS_DIR}/stats/stddev_by_level.nc"
)

if (( ${#params[@]} != 1 )); then
  echo "ERROR: expected one GraphCast_small checkpoint under ${ASSETS_DIR}/params." >&2
  echo "Run: python ${SCRIPT_DIR}/download_assets.py --assets-dir ${ASSETS_DIR}" >&2
  exit 2
fi
if [[ -z "${DATASET}" ]] && (( ${#datasets[@]} != 1 )); then
  echo "ERROR: expected one compatible ERA5 dataset under ${ASSETS_DIR}/dataset." >&2
  echo "Run: python ${SCRIPT_DIR}/download_assets.py --assets-dir ${ASSETS_DIR}" >&2
  exit 2
fi
for file in "${stats[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: missing normalization statistics: ${file}" >&2
    exit 2
  fi
done

log_file="${LOGS_DIR}/inference_${timestamp}.log"
command=(
  python "${SCRIPT_DIR}/run_inference.py"
  --assets-dir "${ASSETS_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --model "${MODEL}"
  --steps "${STEPS}"
  --log-file "${LOGS_DIR}/run_inference_${timestamp}.log"
)
if [[ -n "${DATASET}" ]]; then
  command+=(--dataset "${DATASET}")
fi

export GRAPHCAST_SOURCE_DIR="${SOURCE_DIR}"
export PYTHONPATH="${SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "Running GraphCast with data root: ${DATA_ROOT}"
echo "Combined stdout/stderr log: ${log_file}"
"${command[@]}" 2>&1 | tee "${log_file}"

if [[ ! -s "${OUTPUT_DIR}/predictions.nc" || ! -s "${OUTPUT_DIR}/metrics.json" ]]; then
  echo "ERROR: inference returned success but required output files are absent or empty." >&2
  exit 3
fi
echo "Verified outputs:"
echo "  ${OUTPUT_DIR}/predictions.nc"
echo "  ${OUTPUT_DIR}/metrics.json"
