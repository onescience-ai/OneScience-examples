import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config
from data.airfrans_dataset import AirfRANSDataPipe
from model.infinity import INFINITY
from eval.metrics import evaluate


def main(config_path="configs/infinity.yaml"):
    cfg = load_config(config_path)
    manifest = cfg["defaults"]["manifest"]
    with open(manifest) as f:
        m = json.load(f)
    test_names = m[cfg["defaults"]["test_split"]]

    device = "cuda" if False else "cpu"
    ckpt_full = torch.load(os.path.join(cfg["defaults"]["out_dir"], "infinity_full.pt"),
                           map_location=device)
    norm = ckpt_full["norm"]

    model = INFINITY(cfg)
    inr_ckpt = torch.load(os.path.join(cfg["defaults"]["out_dir"], "infinity_inr.pt"),
                          map_location=device)
    model.inrs.load_state_dict(inr_ckpt["inr_state"])
    model.g_psi.load_state_dict(ckpt_full["g_psi_state"])
    model.eval()

    pipe = AirfRANSDataPipe(cfg["defaults"]["data_root"], test_names)
    res = evaluate(model, pipe, norm, cfg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    import torch
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/infinity.yaml"
    main(cfg)
