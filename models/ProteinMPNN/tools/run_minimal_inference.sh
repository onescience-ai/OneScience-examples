#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/modelscope_minimal

python protein_mpnn_run.py \
  --pdb_path inputs/PDB_monomers/pdbs/5L33.pdb \
  --pdb_path_chains A \
  --path_to_model_weights vanilla_model_weights \
  --model_name v_48_020 \
  --out_folder outputs/modelscope_minimal \
  --num_seq_per_target 2 \
  --sampling_temp "0.1" \
  --seed 37 \
  --batch_size 1
