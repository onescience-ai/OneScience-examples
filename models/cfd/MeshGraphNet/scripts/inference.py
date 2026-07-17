import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from model.meshgraphnet import MeshGraphNet
from onescience.utils.YParams import YParams
from onescience.launch.utils import load_checkpoint 
from fake_data import build_cylinder_flow_datapipe


def build_model(model_params, device):
    mlp_act = "silu" if model_params.recompute_activation else "relu"
    return MeshGraphNet(
        input_dim_nodes=model_params.num_input_features,
        input_dim_edges=model_params.num_edge_features,
        output_dim=model_params.num_output_features,
        processor_size=model_params.processor_size,
        hidden_dim_processor=model_params.hidden_dim_processor,
        num_layers_node_processor=model_params.num_layers_node_processor,
        num_layers_edge_processor=model_params.num_layers_edge_processor,
        hidden_dim_node_encoder=model_params.hidden_dim_node_encoder,
        hidden_dim_edge_encoder=model_params.hidden_dim_edge_encoder,
        hidden_dim_node_decoder=model_params.hidden_dim_node_decoder,
        mlp_activation_fn=mlp_act,
        do_concat_trick=model_params.do_concat_trick,
        num_processor_checkpoint_segments=model_params.num_processor_checkpoint_segments,
        recompute_activation=model_params.recompute_activation,
    ).to(device)


def resolve_device(device_name: str):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested cuda device, but torch.cuda.is_available() is false.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    os.chdir(PROJECT_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("mesh_graph_net.inference")

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    cfg_model = YParams(config_path, "model")
    cfg_data = YParams(config_path, "datapipe")
    cfg_train = YParams(config_path, "training")
    cfg_inference = YParams(config_path, "inference")
    model_params = cfg_model.specific_params[cfg_model.name]

    device = resolve_device(getattr(cfg_inference, "device", "auto"))
    logger.info("Using device: %s", device)
    datapipe = build_cylinder_flow_datapipe(
        params=cfg_data,
        distributed=False,
        project_root=PROJECT_ROOT,
    )
    loader = datapipe.test_dataloader()
    model = build_model(model_params, device)
    checkpoint_dir = PROJECT_ROOT / getattr(cfg_inference, "checkpoint_dir", cfg_train.checkpoint_dir)
    epoch = load_checkpoint(checkpoint_dir, models=model, device=device)
    if epoch == 0:
        logger.warning("No checkpoint found in %s; running with randomly initialized weights", checkpoint_dir)
    model.eval()

    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            graph = batch[0] if isinstance(batch, (tuple, list)) else batch
            graph = graph.to(device)
            pred = model(graph.ndata["x"], graph.edata["x"], graph)
            predictions.append(pred.cpu().numpy())
            targets.append(graph.ndata["y"].cpu().numpy())

    output_path = PROJECT_ROOT / cfg_inference.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, prediction=np.concatenate(predictions, axis=0), target=np.concatenate(targets, axis=0))
    logger.info("Saved inference results to %s", output_path)


if __name__ == "__main__":
    main()
