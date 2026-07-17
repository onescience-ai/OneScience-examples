import json
from pathlib import Path

import numpy as np

from common import load_config


def bytes_feature(value: bytes):
    import tensorflow.compat.v1 as tf

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value: int):
    import tensorflow.compat.v1 as tf

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_positions(rng, num_steps: int, num_particles: int, dim: int, bounds):
    low, high = float(bounds[0]), float(bounds[1])
    position = rng.uniform(low + 0.2, high - 0.2, size=(num_particles, dim))
    frames = []

    for _ in range(num_steps):
        velocity = rng.normal(0.0, 0.015, size=(num_particles, dim))
        position = np.clip(position + velocity, low + 0.05, high - 0.05)
        frames.append(position.astype(np.float32))

    return np.stack(frames, axis=0)


def make_particle_types(num_particles: int, num_node_types: int):
    particle_type = np.full((num_particles,), 5, dtype=np.int64)
    particle_type[::5] = 3
    particle_type = np.clip(particle_type, 0, num_node_types - 1)
    return particle_type


def write_split(path: Path, split: str, num_sequences: int, cfg, rng) -> None:
    import tensorflow.compat.v1 as tf

    num_steps = int(cfg.data.fake.num_steps)
    num_particles = int(cfg.data.fake.num_particles)
    dim = int(cfg.dim)
    bounds = cfg.data.fake.bounds

    writer_path = path / f"{split}.tfrecord"
    with tf.io.TFRecordWriter(str(writer_path)) as writer:
        for index in range(num_sequences):
            positions = make_positions(rng, num_steps, num_particles, dim, bounds)
            particle_type = make_particle_types(num_particles, int(cfg.data.num_node_types))

            context = tf.train.Features(
                feature={
                    "key": int64_feature(index),
                    "particle_type": bytes_feature(particle_type.tobytes()),
                }
            )
            feature_list = tf.train.FeatureList(
                feature=[bytes_feature(frame.tobytes()) for frame in positions]
            )
            example = tf.train.SequenceExample(
                context=context,
                feature_lists=tf.train.FeatureLists(
                    feature_list={"position": feature_list}
                ),
            )
            writer.write(example.SerializeToString())


def write_metadata(path: Path, cfg) -> None:
    dim = int(cfg.dim)
    bounds = [float(v) for v in cfg.data.fake.bounds]
    metadata = {
        "sequence_length": int(cfg.data.fake.num_steps) - 1,
        "dt": 1.0,
        "default_connectivity_radius": float(cfg.data.fake.connectivity_radius),
        "bounds": [bounds for _ in range(dim)],
        "dim": dim,
        "vel_mean": [0.0] * dim,
        "vel_std": [1.0] * dim,
        "acc_mean": [0.0] * dim,
        "acc_std": [1.0] * dim,
    }
    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    cfg = load_config()
    data_dir = Path(cfg.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(data_dir, cfg)

    rng = np.random.default_rng(int(cfg.data.fake.seed))
    for split in ("train", "valid", "test"):
        split_cfg = getattr(cfg.data, split)
        write_split(data_dir, split_cfg.split, int(split_cfg.num_sequences), cfg, rng)

    print(f"Fake Lagrangian TFRecord data written to {data_dir}")


if __name__ == "__main__":
    main()
