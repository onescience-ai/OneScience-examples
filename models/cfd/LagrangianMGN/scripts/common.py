import sys
from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def project_path(path_value) -> str:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def load_config():
    cfg = OmegaConf.merge(OmegaConf.load(CONFIG_PATH), OmegaConf.from_cli())

    cfg.data.data_dir = project_path(cfg.data.data_dir)
    cfg.datapipe.source.data_dir = cfg.data.data_dir
    cfg.output = project_path(cfg.output)
    cfg.resume_dir = project_path(cfg.resume_dir)
    cfg.inference.output_dir = project_path(cfg.inference.output_dir)

    fill_model_dimensions(cfg)
    OmegaConf.resolve(cfg)
    return cfg


def fill_model_dimensions(cfg) -> None:
    dim = int(cfg.dim)
    num_history = int(cfg.data.num_history)
    num_node_types = int(cfg.data.num_node_types)

    if cfg.model.input_dim_nodes is None:
        cfg.model.input_dim_nodes = dim + dim * num_history + 2 * dim + num_node_types
    if cfg.model.input_dim_edges is None:
        cfg.model.input_dim_edges = dim + 1
    if cfg.model.output_dim is None:
        cfg.model.output_dim = dim
