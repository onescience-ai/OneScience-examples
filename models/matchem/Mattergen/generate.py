import argparse
import json
import os
from pathlib import Path

from model.common.utils.data_classes import MatterGenCheckpointInfo
from model.generator import CrystalGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate crystals with MatterGen")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/mattergen")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument(
        "--properties",
        type=json.loads,
        default=None,
        help='Condition values as JSON, for example {"dft_mag_density": 0.15}.',
    )
    parser.add_argument("--record-trajectories", action="store_true")
    args = parser.parse_args()
    if not os.path.isdir(args.checkpoint):
        parser.error(f"checkpoint directory does not exist: {args.checkpoint}")
    checkpoint_info = MatterGenCheckpointInfo(
        model_path=Path(args.checkpoint).expanduser().resolve(),
        load_epoch="last",
    )
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        properties_to_condition_on=args.properties,
        record_trajectories=args.record_trajectories,
    )
    structures = generator.generate(
        output_dir=Path(args.output).expanduser().resolve()
    )
    print(f"Generated {len(structures)} structures in {args.output}")


if __name__ == "__main__":
    main()
