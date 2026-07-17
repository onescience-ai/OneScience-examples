import os
import glob
import yaml


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def find_case_dirs(data_root):
    return sorted(
        d for d in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(d)
    )


def split_cases(manifest_path, train_split, test_split):
    import json
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    train = manifest.get(train_split, [])
    test = manifest.get(test_split, [])
    return train, test
