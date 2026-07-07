import logging
import os
import re

from ruamel.yaml import YAML


def parse_env_vars(data):
    if isinstance(data, str):
        env_var_pattern = r"\$\{(\w+)\}|\$(\w+)"
        return re.sub(
            env_var_pattern,
            lambda m: os.environ.get(m.group(1) or m.group(2), ""),
            data,
        )
    if isinstance(data, dict):
        return {key: parse_env_vars(value) for key, value in data.items()}
    if isinstance(data, list):
        return [parse_env_vars(item) for item in data]
    return data


class AttrDict(dict):
    """Dictionary with recursive dot access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            self[key] = self._wrap(value)

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"No such attribute: {key}") from exc

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __delattr__(self, key):
        del self[key]

    def _wrap(self, value):
        if isinstance(value, dict):
            return AttrDict(value)
        if isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value

    def to_dict(self):
        plain_dict = {}
        for key, value in self.items():
            if isinstance(value, AttrDict):
                plain_dict[key] = value.to_dict()
            elif isinstance(value, list):
                plain_dict[key] = self._convert_list(value)
            else:
                plain_dict[key] = value
        return plain_dict

    def _convert_list(self, values):
        converted = []
        for item in values:
            if isinstance(item, AttrDict):
                converted.append(item.to_dict())
            elif isinstance(item, list):
                converted.append(self._convert_list(item))
            else:
                converted.append(item)
        return converted


class YParams:
    """YAML section parser with recursive dot access."""

    def __init__(self, yaml_filename, config_name, print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name

        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename) as yaml_file:
            data = parse_env_vars(YAML().load(yaml_file))
            config_data = AttrDict(data[config_name])
            self.params = config_data
            for key, value in config_data.items():
                object.__setattr__(self, key, value)
                if print_params:
                    print(f"{key}: {value}")

        if print_params:
            print("---------------------------------------------------")

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value
        object.__setattr__(self, key, self.params[key])

    def __contains__(self, key):
        return key in self.params

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def get(self, key, default=None):
        return self.params.get(key, default)

    def keys(self):
        return self.params.keys()

    def items(self):
        return self.params.items()

    def values(self):
        return self.params.values()

    def update_params(self, config):
        for key, value in config.items():
            self[key] = value

    def to_dict(self):
        return self.params.to_dict()

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: %s", self._yaml_filename)
        logging.info("Configuration name: %s", self._config_name)
        for key, value in self.params.items():
            logging.info("%s: %s", key, value)
        logging.info("---------------------------------------------------")
