#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ONESCIENCE_ROOT="${ONESCIENCE_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}/model:${ONESCIENCE_ROOT}/src:${PYTHONPATH:-}"

folder_with_pdbs="${PROJECT_ROOT}/data/inputs/PDB_monomers/pdbs/"

output_dir="${PROJECT_ROOT}/outputs/example_1_outputs"
if [ ! -d "$output_dir" ]
then
    mkdir -p "$output_dir"
fi

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"

python "${PROJECT_ROOT}/scripts/helper_scripts/parse_multiple_chains.py" --input_path="$folder_with_pdbs" --output_path="$path_for_parsed_chains"

python "${PROJECT_ROOT}/scripts/inference.py" \
        --jsonl_path "$path_for_parsed_chains" \
        --out_folder "$output_dir" \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1 \
        --path_to_model_weights "${PROJECT_ROOT}/weight/vanilla_model_weights"
