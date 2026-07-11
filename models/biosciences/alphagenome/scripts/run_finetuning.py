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

"""AlphaGenome finetuning example script.

This script demonstrates how to finetune the AlphaGenome model on custom genomic data, suitable for the following scenarios:
  - ATAC-seq/ChIP-seq signal prediction for new cell types or tissues
  - Adaptation to specific experimental data
  - Transfer learning to new species

Data requirements:
  - Reference genome FASTA file
  - BigWig signal track files pointed to by file_path in metadata
  - Training regions CSV file (columns: chromosome, start, end)

Usage:
    python run_finetuning.py \
        --fasta_path /path/to/GRCh38.fa \
        --regions_csv /path/to/regions.csv \
        --output_dir ./finetuned_model \
        --num_steps 1000 \
        --batch_size 2
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

from alphagenome._sdk.data import fold_intervals
from alphagenome._sdk.models import dna_model as dna_model_types
import jax
import optax
import orbax.checkpoint as ocp

from alphagenome.finetuning.finetune import (
    get_dataset_iterator,
    get_forward_fn,
    get_train_step,
)
from alphagenome.evals.track_prediction import load_model as load_model_from_kaggle
from alphagenome.model.metadata import metadata as metadata_lib

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'fasta_path',
    None,
    'Reference genome FASTA path.',
    required=True,
)
flags.DEFINE_string(
    'regions_csv',
    None,
    'Training regions CSV path with chromosome,start,end columns.',
    required=True,
)
flags.DEFINE_list(
    'bigwig_paths',
    None,
    'Deprecated compatibility flag. BigWig paths are read from metadata '
    'file_path columns.',
)
flags.DEFINE_string(
    'model_dir',
    None,
    'Pretrained checkpoint directory. If unset, Kaggle Hub is used.',
)
flags.DEFINE_string(
    'output_dir',
    './finetuned_model',
    'Directory to save finetuned checkpoints.',
)
flags.DEFINE_integer('num_steps', 1000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 2, 'Training batch size.')
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')
flags.DEFINE_integer('log_every', 50, 'Log interval in steps.')
flags.DEFINE_integer('save_every', 200, 'Checkpoint interval in steps.')
flags.DEFINE_enum(
    'model_version', 'FOLD_0',
    ['FOLD_0', 'FOLD_1', 'FOLD_2', 'FOLD_3', 'FOLD_4'],
    'Pretrained model version.',
)
flags.DEFINE_enum(
    'organism', 'HOMO_SAPIENS',
    ['HOMO_SAPIENS', 'MUS_MUSCULUS'],
    'Target organism.',
)


def _resolve_local_model_dir(path: str) -> pathlib.Path:
    model_dir = pathlib.Path(path).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f'Pretrained checkpoint directory does not exist: {model_dir}'
        )
    return model_dir


def _load_pretrained_state(model_version: dna_model_types.ModelVersion):
    if FLAGS.model_dir:
        checkpoint_path = _resolve_local_model_dir(FLAGS.model_dir)
        logging.info('Loading pretrained model from local checkpoint: %s',
                     checkpoint_path)
        return ocp.StandardCheckpointer().restore(str(checkpoint_path))

    logging.info('Loading pretrained model from Kaggle Hub: %s',
                 model_version.name)
    params, state, _ = load_model_from_kaggle(model_version)
    return params, state


def main(_):
    output_dir = pathlib.Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_version = dna_model_types.ModelVersion[FLAGS.model_version]
    organism = dna_model_types.Organism[FLAGS.organism]

    logging.info('JAX devices: %s', jax.devices())
    logging.info('Finetuning config: lr=%.2e, steps=%d, batch_size=%d',
                 FLAGS.learning_rate, FLAGS.num_steps, FLAGS.batch_size)

    # Load pretrained model parameters, the training forward function reconstructs the loss from the finetuning module.
    params, state = _load_pretrained_state(model_version)

    # Load output metadata.
    output_metadata = metadata_lib.load(organism)

    # Build the optimizer, using warmup + cosine decay to balance stability and convergence.
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=FLAGS.learning_rate,
        warmup_steps=100,
        decay_steps=FLAGS.num_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule),
    )
    opt_state = optimizer.init(params)

    # Build finetuning training steps.
    forward = get_forward_fn({organism: output_metadata})
    train_step = get_train_step(
        predict_fn=forward.apply,
        optimizer=optimizer,
    )

    # Build dataset iterator.
    logging.info('Building finetuning dataset iterator...')
    dataset_iter = get_dataset_iterator(
        batch_size=FLAGS.batch_size,
        sequence_length=1_048_576,
        output_metadata=output_metadata,
        model_version=model_version,
        subset=fold_intervals.Subset.TRAIN,
        organism=organism,
        fasta_path=FLAGS.fasta_path,
        example_regions_path=FLAGS.regions_csv,
    )

    # Configure checkpoint manager.
    checkpointer = ocp.CheckpointManager(
        output_dir / 'checkpoints',
        options=ocp.CheckpointManagerOptions(max_to_keep=3),
    )

    # Training loop.
    logging.info('Starting finetuning training...')
    for step, batch in enumerate(dataset_iter):
        if step >= FLAGS.num_steps:
            break

        params, state, opt_state, metrics = train_step(
            params, state, opt_state, batch
        )

        if step % FLAGS.log_every == 0:
            loss = float(metrics.get('loss', float('nan')))
            logging.info('Step %d/%d | loss=%.4f', step, FLAGS.num_steps, loss)

        if step % FLAGS.save_every == 0 and step > 0:
            checkpointer.save(step, args=ocp.args.StandardSave({'params': params, 'state': state}))
            logging.info('Checkpoint saved (step=%d)', step)

    # Save final model.
    checkpointer.save(
        FLAGS.num_steps,
        args=ocp.args.StandardSave({'params': params, 'state': state}),
    )
    logging.info('Finetuning complete, final model saved to %s', output_dir / 'checkpoints')


if __name__ == '__main__':
    app.run(main)