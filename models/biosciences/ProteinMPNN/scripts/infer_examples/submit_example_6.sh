#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ONESCIENCE_ROOT="${ONESCIENCE_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}/model:${ONESCIENCE_ROOT}/src:${PYTHONPATH:-}"

folder_with_pdbs="${PROJECT_ROOT}/data/inputs/PDB_homooligomers/pdbs/"

output_dir="${PROJECT_ROOT}/outputs/example_6_outputs"
if [ ! -d "$output_dir" ]
then
    mkdir -p "$output_dir"
fi


path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_tied_positions=$output_dir"/tied_pdbs.jsonl"
path_for_designed_sequences=$output_dir"/temp_0.1"

python "${PROJECT_ROOT}/scripts/helper_scripts/parse_multiple_chains.py" --input_path="$folder_with_pdbs" --output_path="$path_for_parsed_chains"

python "${PROJECT_ROOT}/scripts/helper_scripts/make_tied_positions_dict.py" --input_path="$path_for_parsed_chains" --output_path="$path_for_tied_positions" --homooligomer 1

python "${PROJECT_ROOT}/scripts/inference.py" \
        --jsonl_path "$path_for_parsed_chains" \
        --tied_positions_jsonl "$path_for_tied_positions" \
        --out_folder "$output_dir" \
        --num_seq_per_target 2 \
        --sampling_temp "0.2" \
        --seed 37 \
        --batch_size 1 \
        --path_to_model_weights "${PROJECT_ROOT}/weight/vanilla_model_weights"
