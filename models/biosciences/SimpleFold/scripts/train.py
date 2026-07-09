#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule
import hydra
from omegaconf import OmegaConf
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onescience.utils.simplefold.utils import (
    extras,
    create_folders,
    task_wrapper,
)
from onescience.utils.simplefold.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_trainer,
)
from onescience.utils.simplefold.logging_utils import log_hyperparameters
from onescience.utils.simplefold.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

ESM2_MODEL_NAME = "esm2_t36_3B_UR50D"

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True # This flag defaults to False
torch.backends.cudnn.allow_tf32 = True       # This flag defaults to True


def configure_local_esm_registry(esm_model: str | None) -> None:
    if esm_model != "esm2_3B":
        return

    model_path = Path(
        os.getenv(
            "SIMPLEFOLD_ESM2_MODEL_PATH",
            ROOT / "weight" / "esm_models" / f"{ESM2_MODEL_NAME}.pt",
        )
    )
    regression_path = model_path.with_name(f"{model_path.stem}-contact-regression.pt")

    if not model_path.exists() or not regression_path.exists():
        raise FileNotFoundError(
            "Local ESM-2 3B weights were not found. Expected files:\n"
            f"  {model_path}\n"
            f"  {regression_path}\n"
            "Set SIMPLEFOLD_ESM2_MODEL_PATH to the local .pt file if it lives elsewhere."
        )

    import models.esm.pretrained as local_esm_pretrained
    import onescience.utils.simplefold.esm_utils as esm_utils

    def load_local_esm2_3b():
        print(f"Loading ESM-2 3B weights from local path: {model_path}")
        return local_esm_pretrained.load_model_and_alphabet(str(model_path))

    esm_utils.esm_registry["esm2_3B"] = load_local_esm2_3b
    simplefold_module = sys.modules.get("models.simplefold.simplefold")
    if simplefold_module is not None:
        simplefold_module.esm_registry["esm2_3B"] = load_local_esm2_3b


@task_wrapper
def train(cfg):
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    configure_local_esm_registry(cfg.model.get("esm_model", None))

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    load_ckpt_path = cfg.get("load_ckpt_path", None)

    if load_ckpt_path is not None:
        # load existing ckpt
        log.info(f"Resuming from checkpoint <{cfg.load_ckpt_path}>...")
        model.strict_loading = False

        # manually reset these variables in case of fine-tuning
        model.lddt_weight_schedule = cfg.model.get("lddt_weight_schedule", False)
        model.plddt_training = cfg.model.get("plddt_training", False)

        # reset ESM model to avoid issues in loading FSDP checkpoint
        model.reset_esm(cfg.model.esm_model)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    OmegaConf.set_struct(cfg.logger, True)
    loggers = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = instantiate_trainer(
        cfg.trainer, callbacks=callbacks, logger=loggers, plugins=None
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if log:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=load_ckpt_path,
    )


@hydra.main(version_base="1.3", config_path="../config", config_name="train.yaml")
def submit_run(cfg):
    OmegaConf.resolve(cfg)
    extras(cfg)
    create_folders(cfg)
    train(cfg)
    return


if __name__ == "__main__":
    submit_run()
