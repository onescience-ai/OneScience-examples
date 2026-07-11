#!/usr/bin/env bash
set -euo pipefail

# Run this script from the ESM project root.

PROJECT_MODEL="OneScience/ESM"

download_project_dir() {
  local remote_path="$1"
  local local_dir="$2"
  mkdir -p "$local_dir"
  modelscope download --model "$PROJECT_MODEL" "$remote_path" --local_dir "$local_dir"
}

download_project_dir "ESM/data" "./data"
download_project_dir "ESM/weight" "./weight"
