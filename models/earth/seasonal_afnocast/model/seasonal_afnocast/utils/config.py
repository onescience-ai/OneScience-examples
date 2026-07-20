"""
Configuration management and parameter handling utilities.

Handles YAML/JSON config loading, command-line argument parsing,
and environment-specific configuration management for reproducible
experiments and deployments.
"""

import yaml
import omegaconf
from pathlib import Path

def load_config(config_path: str):
    """
    Load configuration from a YAML file and convert to OmegaConf format.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        # resolve relative to the repo root (or a known anchor dir)
        config_path = Path(__file__).resolve().parent.parent.parent.parent / config_path
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = omegaconf.OmegaConf.create(config)
    return config

def save_config(config, config_path: str):
    """
    Save configuration to a YAML file from OmegaConf format.
    """
    with open(config_path, "w") as file:
        yaml.dump(omegaconf.OmegaConf.to_container(config), file)

def update_config(config, updates: dict):
    """
    Update configuration with a dictionary of new values.
    """
    for key, value in updates.items():
        if isinstance(value, dict):
            config[key] = omegaconf.OmegaConf.create(value)
        else:
            config[key] = value
    return config

if __name__ == "__main__":
    config = load_config("configs/training_config.yaml")
    print(config)