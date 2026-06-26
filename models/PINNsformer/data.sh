#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATASETS_DIR="/public/share/sugonhpcapp01/onestore/onedatasets"
DATASETS_DIR="${ONESCIENCE_DATASETS_DIR:-${DEFAULT_DATASETS_DIR}}"
SOURCE_DIR="${DATASETS_DIR}/pinnsformer"

if [ ! -d "${SOURCE_DIR}" ]; then
  echo "source data not found: ${SOURCE_DIR}" >&2
  echo "Set ONESCIENCE_DATASETS_DIR to a directory containing pinnsformer/convection.mat and pinnsformer/cylinder_nektar_wake.mat." >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/convection" "${SCRIPT_DIR}/navier_stokes"
cp "${SOURCE_DIR}/convection.mat" "${SCRIPT_DIR}/convection/convection.mat"
cp "${SOURCE_DIR}/cylinder_nektar_wake.mat" "${SCRIPT_DIR}/navier_stokes/cylinder_nektar_wake.mat"

echo "[OK] copied PINNsformer data from ${SOURCE_DIR}"
