"""Fine-tuning helpers backed by the example's local MatterGen model package."""

import logging
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, open_dict

from .common.utils.data_classes import MatterGenCheckpointInfo, _rewrite_vendored_targets
from .common.utils.globals import get_device

logger = logging.getLogger(__name__)


def init_adapter_lightningmodule_from_pretrained(
    adapter_cfg: DictConfig, lightning_module_cfg: DictConfig
) -> Tuple[pl.LightningModule, DictConfig]:
    """Initialize the local adapter model from a MatterGen checkpoint."""
    if adapter_cfg.model_path is not None:
        if adapter_cfg.pretrained_name is not None:
            logger.warning(
                "pretrained_name is provided, but will be ignored since model_path is also provided."
            )
        model_path = Path(hydra.utils.to_absolute_path(adapter_cfg.model_path))
        checkpoint_info = MatterGenCheckpointInfo(model_path, adapter_cfg.load_epoch)
    elif adapter_cfg.pretrained_name is not None:
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
            adapter_cfg.pretrained_name
        )
    else:
        raise ValueError("Either adapter.model_path or adapter.pretrained_name is required.")

    checkpoint_path = checkpoint_info.checkpoint_path
    version_root_path = Path(checkpoint_path).relative_to(
        checkpoint_info.model_path
    ).parents[1]
    config_path = Path(checkpoint_info.model_path) / version_root_path
    pretrained_cfg_path = (
        config_path if (config_path / "config.yaml").exists() else config_path.parent.parent
    )

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        str(pretrained_cfg_path.absolute()), version_base="1.1"
    ):
        pretrained_cfg = hydra.compose(config_name="config")
    pretrained_cfg = _rewrite_vendored_targets(pretrained_cfg)
    diffusion_module_cfg = deepcopy(pretrained_cfg.lightning_module.diffusion_module)
    denoiser_cfg = diffusion_module_cfg.model

    with open_dict(adapter_cfg.adapter):
        for key, value in denoiser_cfg.items():
            if key not in {"_target_", "property_embeddings_adapt"}:
                adapter_cfg.adapter[key] = value
            if key == "property_embeddings":
                for field in value:
                    if field in adapter_cfg.adapter.property_embeddings_adapt:
                        adapter_cfg.adapter.property_embeddings_adapt.remove(field)

        adapter_cfg.adapter.gemnet["_target_"] = (
            "model.common.gemnet.gemnet_ctrl.GemNetTCtrl"
        )
        adapter_cfg.adapter.gemnet.condition_on_adapt = list(
            adapter_cfg.adapter.property_embeddings_adapt
        )

    with open_dict(diffusion_module_cfg):
        diffusion_module_cfg.model = adapter_cfg.adapter
    with open_dict(lightning_module_cfg):
        lightning_module_cfg.diffusion_module = diffusion_module_cfg

    lightning_module = hydra.utils.instantiate(lightning_module_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    pretrained_state: OrderedDict = checkpoint["state_dict"]
    local_state: OrderedDict = lightning_module.state_dict()
    local_state.update(
        (key, pretrained_state[key])
        for key in local_state.keys() & pretrained_state.keys()
    )
    lightning_module.load_state_dict(local_state, strict=True)

    if not adapter_cfg.full_finetuning:
        pretrained_keys = set(pretrained_state.keys())
        for name, parameter in lightning_module.named_parameters():
            if name in pretrained_keys:
                parameter.requires_grad_(False)

    return lightning_module, lightning_module_cfg
