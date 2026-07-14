#!/usr/bin/env bash
set -Eeuo pipefail

# TargetDiff repository root (this script should be placed in the repository root).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Override when necessary, for example:
# MODELSCOPE_MODEL=your_name/your_model bash download_assets.sh
MODELSCOPE_MODEL="${MODELSCOPE_MODEL:-OneScience/TargetDiff}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ASSET_PATHS=(
  "examples/3ug2_ligand.sdf"
  "examples/3ug2_protein.pdb"
  "weight/egnn_pdbbind_v2016.pt"
  "weight/pk_reg_para.pkl"
  "weight/pretrained_diffusion.pt"
)

echo "== TargetDiff assets download =="
echo "Repository root: ${ROOT_DIR}"
echo "ModelScope model: ${MODELSCOPE_MODEL}"

if ! command -v modelscope >/dev/null 2>&1; then
  echo "ModelScope CLI was not found; installing modelscope..."
  "${PYTHON_BIN}" -m pip install modelscope
  hash -r
fi

if ! command -v modelscope >/dev/null 2>&1; then
  echo "ERROR: modelscope was installed, but its executable is not in PATH." >&2
  echo "Add the Python scripts/bin directory to PATH and rerun this script." >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/examples" "${ROOT_DIR}/weight"

for asset_path in "${ASSET_PATHS[@]}"; do
  echo "Downloading ${asset_path}"
  modelscope download \
    --model "${MODELSCOPE_MODEL}" \
    "${asset_path}" \
    --local_dir "${ROOT_DIR}"
done

missing_files=()
for asset_path in "${ASSET_PATHS[@]}"; do
  if [[ ! -s "${ROOT_DIR}/${asset_path}" ]]; then
    missing_files+=("${asset_path}")
  fi
done

if (( ${#missing_files[@]} > 0 )); then
  echo "ERROR: The following assets are missing or empty after download:" >&2
  printf '  - %s\n' "${missing_files[@]}" >&2
  exit 1
fi

echo "All TargetDiff assets were downloaded successfully."
