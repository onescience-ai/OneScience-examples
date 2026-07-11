#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ONESCIENCE_ROOT="${ONESCIENCE_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}/model:${ONESCIENCE_ROOT}/src:${PYTHONPATH:-}"

path_to_PDB="${PROJECT_ROOT}/data/inputs/PDB_complexes/pdbs/3HTN.pdb"
path_to_fasta="${PROJECT_ROOT}/outputs/example_3_outputs/seqs/3HTN.fa"

output_dir="${PROJECT_ROOT}/outputs/example_3_score_only_from_fasta_outputs"
if [ ! -d "$output_dir" ]
then
    mkdir -p "$output_dir"
fi

chains_to_design="A B"

python "${PROJECT_ROOT}/scripts/inference.py" \
        --path_to_fasta "$path_to_fasta" \
        --pdb_path "$path_to_PDB" \
        --pdb_path_chains "$chains_to_design" \
        --out_folder "$output_dir" \
        --num_seq_per_target 5 \
        --sampling_temp "0.1" \
        --score_only 1 \
        --seed 13 \
        --batch_size 1 \
        --path_to_model_weights "${PROJECT_ROOT}/weight/vanilla_model_weights"
