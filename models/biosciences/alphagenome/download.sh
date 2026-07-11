#!/usr/bin/env bash
set -euo pipefail

# Run this script from the alphagenome project root.

PROJECT_MODEL="OneScience/alphagenome"
DATASET_MODEL="OneScience/alphagenome_dataset"

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

download_project_dir "alphagenome/weight" "./weight"
download_project_dir "alphagenome/flax_model/alphagenome/model/metadata/OutputMetadataResponse_ORGANISM_HOMO_SAPIENS.textproto" "./flax_model/alphagenome/model/metadata"
download_project_dir "alphagenome/flax_model/alphagenome/model/metadata/OutputMetadataResponse_ORGANISM_MUS_MUSCULUS.textproto" "./flax_model/alphagenome/model/metadata"
download_model_repo "$DATASET_MODEL" "./alphagenome_dataset"
