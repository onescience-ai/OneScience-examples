#!/usr/bin/env bash
set -euo pipefail

# Run this script from the AlphaFold3 project root.

PROJECT_MODEL="OneScience/AlphaFold3"
DATASET_MODEL="OneScience/AlphaFold3_dataset"

download_project_dir() {
  local remote_path="$1"
  local local_dir="$2"
  mkdir -p "$local_dir"
  modelscope download --model "$PROJECT_MODEL" "$remote_path" --local_dir "$local_dir"
}

download_model_repo() {
  local model_id="$1"
  local local_dir="$2"
  mkdir -p "$local_dir"
  modelscope download --model "$model_id" --local_dir "$local_dir"
}

download_project_dir "AlphaFold3/flax_model/alphafold3/test_data" "./flax_model/alphafold3/test_data"
download_project_dir "AlphaFold3/inputs" "./inputs"
download_project_dir "AlphaFold3/weight" "./weight"

download_model_repo "$DATASET_MODEL" "./AlphaFold3_dataset"
