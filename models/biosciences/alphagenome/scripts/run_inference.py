# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AlphaGenome inference example script.

This script demonstrates how to use AlphaGenome for multimodal prediction on genomic intervals, including:
  - Gene expression (RNA-seq, CAGE)
  - Chromatin accessibility (ATAC-seq, DNase-seq)
  - Transcription factor binding (ChIP-seq)
  - Hi-C contact maps
  - Splice sites

Usage:
    # Automatically download the model using Kaggle Hub (default mode)
    python run_inference.py

    # Specify a local reference genome
    python run_inference.py \
        --fasta_path /path/to/GRCh38.fa \
        --chromosome chr19 \
        --start 10587331 \
        --end 11635907 \
        --output_dir ./outputs
"""

import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from absl import app
from absl import flags
from absl import logging

from alphagenome._sdk.data import genome
from alphagenome.model import dna_model as dna_model_types
import numpy as np

from alphagenome.model.dna_model import (
    create,
    create_from_kaggle,
    OrganismSettings,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'fasta_path',
    None,
    'Reference genome FASTA path. A .fai index is required.',
)
flags.DEFINE_string(
    'model_dir',
    None,
    'Local AlphaGenome checkpoint directory. If unset, Kaggle Hub is used.',
)
flags.DEFINE_string('chromosome', 'chr1', 'Chromosome name, for example chr1.')
flags.DEFINE_integer('start', 1_000_000, 'Interval start, 0-based.')
flags.DEFINE_integer('end', 2_048_576, 'Interval end, exclusive.')
flags.DEFINE_string('output_dir', './outputs', 'Output directory.')
flags.DEFINE_enum(
    'organism', 'HOMO_SAPIENS',
    ['HOMO_SAPIENS', 'MUS_MUSCULUS'],
    'Target organism.',
)
flags.DEFINE_enum(
    'model_version', 'FOLD_0',
    ['FOLD_0', 'FOLD_1', 'FOLD_2', 'FOLD_3', 'FOLD_4', 'all_folds'],
    'Model version, fold, or all_folds ensemble.',
)


def main(_):
    output_dir = pathlib.Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    organism = dna_model_types.Organism[FLAGS.organism]
    model_version = FLAGS.model_version
    interval = genome.Interval(
        chromosome=FLAGS.chromosome,
        start=FLAGS.start,
        end=FLAGS.end,
    )

    logging.info('Loading model weights...')
    if FLAGS.model_dir:
        organism_settings = None
        if FLAGS.fasta_path:
            organism_settings = {
                organism: OrganismSettings(
                    fasta_path=FLAGS.fasta_path,
                ),
            }
        alphagenome_model = create(
            checkpoint_path=FLAGS.model_dir,
            organism_settings=organism_settings,
        )
    else:
        alphagenome_model = create_from_kaggle(model_version)

    logging.info('Running inference: %s', interval)
    if FLAGS.fasta_path or not FLAGS.model_dir:
        predictions = alphagenome_model.predict_interval(
            interval,
            organism=organism,
            requested_outputs={
                dna_model_types.OutputType.ATAC,
                dna_model_types.OutputType.DNASE,
                dna_model_types.OutputType.CAGE,
                dna_model_types.OutputType.RNA_SEQ,
                dna_model_types.OutputType.CHIP_TF,
                dna_model_types.OutputType.CHIP_HISTONE,
            },
            ontology_terms=None,
        )
    else:
        logging.warning(
            'No fasta_path was provided; using a random sequence for demo.'
        )
        rng = np.random.default_rng(42)
        bases = np.array(['A', 'C', 'G', 'T'])
        seq_len = interval.end - interval.start
        dna_sequence = ''.join(rng.choice(bases, size=seq_len))
        predictions = alphagenome_model.predict_sequence(
            dna_sequence,
            organism=organism,
            requested_outputs={
                dna_model_types.OutputType.ATAC,
                dna_model_types.OutputType.DNASE,
                dna_model_types.OutputType.RNA_SEQ,
            },
            ontology_terms=None,
            interval=interval,
        )

    logging.info('Saving predictions to %s', output_dir)
    for attr_name in ['atac', 'dnase', 'cage', 'rna_seq', 'chip_tf',
                       'chip_histone', 'contact_maps', 'procap']:
        output = getattr(predictions, attr_name, None)
        if output is not None:
            output_path = output_dir / f'{attr_name}.npy'
            np.save(output_path, np.array(output.values))
            logging.info('  saved %s: shape=%s', attr_name,
                         np.array(output.values).shape)

    logging.info('Inference finished.')


if __name__ == '__main__':
    app.run(main)