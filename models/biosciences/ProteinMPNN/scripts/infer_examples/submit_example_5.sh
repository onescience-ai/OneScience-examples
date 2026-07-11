#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ONESCIENCE_ROOT="${ONESCIENCE_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}/model:${ONESCIENCE_ROOT}/src:${PYTHONPATH:-}"

folder_with_pdbs="${PROJECT_ROOT}/data/inputs/PDB_complexes/pdbs/"

output_dir="${PROJECT_ROOT}/outputs/example_5_outputs"
if [ ! -d "$output_dir" ]
then
    mkdir -p "$output_dir"
fi


path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_assigned_chains=$output_dir"/assigned_pdbs.jsonl"
path_for_fixed_positions=$output_dir"/fixed_pdbs.jsonl"
path_for_tied_positions=$output_dir"/tied_pdbs.jsonl"
chains_to_design="A C"
fixed_positions="9 10 11 12 13 14 15 16 17 18 19 20 21 22 23, 10 11 18 19 20 22"
tied_positions="1 2 3 4 5 6 7 8, 1 2 3 4 5 6 7 8" #two list must match in length; residue 1 in chain A and C will be sampled togther;

python "${PROJECT_ROOT}/scripts/helper_scripts/parse_multiple_chains.py" --input_path="$folder_with_pdbs" --output_path="$path_for_parsed_chains"

python "${PROJECT_ROOT}/scripts/helper_scripts/assign_fixed_chains.py" --input_path="$path_for_parsed_chains" --output_path="$path_for_assigned_chains" --chain_list "$chains_to_design"

python "${PROJECT_ROOT}/scripts/helper_scripts/make_fixed_positions_dict.py" --input_path="$path_for_parsed_chains" --output_path="$path_for_fixed_positions" --chain_list "$chains_to_design" --position_list "$fixed_positions"

python "${PROJECT_ROOT}/scripts/helper_scripts/make_tied_positions_dict.py" --input_path="$path_for_parsed_chains" --output_path="$path_for_tied_positions" --chain_list "$chains_to_design" --position_list "$tied_positions"

python "${PROJECT_ROOT}/scripts/inference.py" \
        --jsonl_path "$path_for_parsed_chains" \
        --chain_id_jsonl "$path_for_assigned_chains" \
        --fixed_positions_jsonl "$path_for_fixed_positions" \
        --tied_positions_jsonl "$path_for_tied_positions" \
        --out_folder "$output_dir" \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1 \
        --path_to_model_weights "${PROJECT_ROOT}/weight/vanilla_model_weights"
