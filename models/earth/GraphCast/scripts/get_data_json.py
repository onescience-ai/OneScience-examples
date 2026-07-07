import os
import json
import sys
from _bootstrap import prepare_runtime

current_path = str(prepare_runtime())

from graphcast_src.utils.YParams import YParams


def main():
    config_file_path = os.path.join(current_path, 'config/config.yaml')
    cfg_data = YParams(config_file_path, "datapipe")
    metadata = {
        "coords": {
            "channel": {str(i): name for i, name in enumerate(cfg_data.dataset.channels)}
        }
    }
    with open('./data.json', "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved.")



if __name__ == '__main__':
    main()
