#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import sys
import torch
import hydra
import omegaconf
import argparse
import numpy as np
from copy import deepcopy
from pathlib import Path
from itertools import starmap
import lightning.pytorch as pl

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
WEIGHT_DIR = ROOT / "weight"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.simplefold.flow import LinearPath
from models.simplefold.torch.sampler import EMSampler

from onescience.datapipes.simplefold.processor.protein_processor import ProteinDataProcessor
from onescience.utils.simplefold.datamodule_utils import process_one_inference_structure
from onescience.utils.simplefold.esm_utils import _af2_to_esm, esm_registry
import models.esm.pretrained as esm_pretrained
from onescience.utils.simplefold.boltz_utils import process_structure, save_structure
from onescience.utils.simplefold.fasta_utils import process_fastas, check_fasta_inputs
from onescience.datapipes.boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from onescience.datapipes.boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer

try: 
    import mlx.core as mx
    from mlx.utils import tree_unflatten, tree_flatten
    from models.simplefold.mlx.sampler import EMSampler as EMSamplerMLX
    from models.simplefold.mlx.esm_network import ESM2 as ESM2MLX
    from onescience.utils.simplefold.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx
    MLX_AVAILABLE = True
except:
    MLX_AVAILABLE = False
    print("MLX not installed, skip importing MLX related packages.")


SUPPORTED_MODELS = {
    "simplefold_100M",
    "simplefold_360M",
    "simplefold_700M",
    "simplefold_1.1B",
    "simplefold_1.6B",
    "simplefold_3B",
}

ESM2_MODEL_NAME = "esm2_t36_3B_UR50D"
MIN_REAL_WEIGHT_BYTES = 1024


def _resolve_path(path: str | os.PathLike[str], base: Path = ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def _require_local_file(path: Path, description: str, min_bytes: int = MIN_REAL_WEIGHT_BYTES) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            f"Please place the file under {WEIGHT_DIR} or pass the corresponding environment/CLI path."
        )
    if path.stat().st_size < min_bytes:
        raise RuntimeError(
            f"{description} looks like a link/placeholder rather than a real weight file: {path}\n"
            "Replace it with the real file before running inference."
        )
    return path


def load_local_or_hub_esm2_3b():
    candidate_paths = []

    esm_model_path = os.getenv("SIMPLEFOLD_ESM2_MODEL_PATH")
    if esm_model_path:
        candidate_paths.append(esm_model_path)

    candidate_paths.extend(
        [
            WEIGHT_DIR / "esm_models" / f"{ESM2_MODEL_NAME}.pt",
            WEIGHT_DIR / f"{ESM2_MODEL_NAME}.pt",
        ]
    )

    seen_paths = set()
    for candidate_path in candidate_paths:
        if candidate_path in seen_paths:
            continue
        seen_paths.add(candidate_path)

        candidate_path = Path(candidate_path)
        regression_path = candidate_path.with_name(
            f"{candidate_path.stem}-contact-regression.pt"
        )
        if candidate_path.exists() and regression_path.exists():
            _require_local_file(candidate_path, "ESM-2 3B model weight")
            _require_local_file(regression_path, "ESM-2 contact regression weight")
            print(f"Loading ESM-2 3B weights from local path: {candidate_path}")
            return esm_pretrained.load_model_and_alphabet(str(candidate_path))

    raise FileNotFoundError(
        "Local ESM-2 3B weights were not found. Put both files here:\n"
        f"  {WEIGHT_DIR / 'esm_models' / f'{ESM2_MODEL_NAME}.pt'}\n"
        f"  {WEIGHT_DIR / 'esm_models' / f'{ESM2_MODEL_NAME}-contact-regression.pt'}\n"
        "or set SIMPLEFOLD_ESM2_MODEL_PATH to the .pt file. Remote download is disabled in this package."
    )


def resolve_ccd_path(cache: Path) -> Path:
    candidate_paths = []

    ccd_path = os.getenv("SIMPLEFOLD_CCD_PATH")
    if ccd_path:
        candidate_paths.append(Path(ccd_path))

    candidate_paths.append(WEIGHT_DIR / "ccd.pkl")

    seen_paths = set()
    for candidate_path in candidate_paths:
        resolved_path = str(candidate_path)
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)

        if candidate_path.exists():
            _require_local_file(candidate_path, "CCD dictionary")
            print(f"Using local CCD dictionary: {candidate_path}")
            return candidate_path

    raise FileNotFoundError(
        f"Missing CCD dictionary. Put ccd.pkl under {WEIGHT_DIR} or set SIMPLEFOLD_CCD_PATH."
    )


def get_config_path(relative_path):
    """Resolve a SimpleFold config from the bundled config directory."""
    config_subpath = relative_path.replace("configs/", "")
    config_path = CONFIG_DIR / config_subpath
    if not config_path.is_file():
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    return str(config_path)



def initialize_folding_model(args):
    # define folding model
    simplefold_model = args.simplefold_model
    if simplefold_model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {simplefold_model!r}. Choose one of {sorted(SUPPORTED_MODELS)}")

    # create checkpoint directory
    ckpt_dir = _resolve_path(args.ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")

    # create folding model
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")
    _require_local_file(Path(ckpt_path), f"{simplefold_model} checkpoint")
    cfg_path = get_config_path(f"configs/model/architecture/foldingdit_{simplefold_model[11:]}.yaml")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # load model checkpoint
    if args.backend == 'torch':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = omegaconf.OmegaConf.load(cfg_path)
        model = hydra.utils.instantiate(model_config)
        model.load_state_dict(checkpoint, strict=True)
        model = model.to(device)
    elif args.backend == 'mlx':
        device = "cpu"
        # replace torch implementations with mlx
        with open(cfg_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        model_config = omegaconf.OmegaConf.create(yaml_str)
        model = hydra.utils.instantiate(model_config)
        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, checkpoint.items()) if k is not None}
        model.update(tree_unflatten(list(mlx_state_dict.items())))
    print(f"Folding model {simplefold_model} loaded.")
    print(f"Using device: {device}.")

    model.eval()
    return model, device


def initialize_plddt_module(args, device):
    if not args.plddt:
        return None, None

    # load pLDDT module if specified
    ckpt_dir = _resolve_path(args.ckpt_dir)
    plddt_ckpt_path = ckpt_dir / "plddt.ckpt"
    if not plddt_ckpt_path.exists():
        plddt_ckpt_path = ckpt_dir / "plddt_module_1.6B.ckpt"
    _require_local_file(plddt_ckpt_path, "pLDDT checkpoint")

    plddt_module_path = get_config_path("configs/model/architecture/plddt_module.yaml")
    plddt_checkpoint = torch.load(plddt_ckpt_path, map_location="cpu", weights_only=False)

    if args.backend == "torch":
        plddt_config = omegaconf.OmegaConf.load(plddt_module_path)
        plddt_out_module = hydra.utils.instantiate(plddt_config)
        plddt_out_module.load_state_dict(plddt_checkpoint, strict=True)
        plddt_out_module = plddt_out_module.to(device)
    elif args.backend == "mlx":
        # replace torch implementations with mlx
        with open(plddt_module_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        plddt_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_out_module = hydra.utils.instantiate(plddt_config)

        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_plddt_torch_to_mlx, plddt_checkpoint.items()) if k is not None}
        plddt_out_module.update(tree_unflatten(list(mlx_state_dict.items())))

    plddt_out_module.eval()
    print(f"pLDDT output module loaded with {args.backend} backend.")

    plddt_latent_ckpt_path = ckpt_dir / "simplefold_1.6B.ckpt"
    _require_local_file(plddt_latent_ckpt_path, "SimpleFold 1.6B pLDDT latent checkpoint")

    plddt_latent_config_path = get_config_path("configs/model/architecture/foldingdit_1.6B.yaml")
    plddt_latent_checkpoint = torch.load(plddt_latent_ckpt_path, map_location="cpu", weights_only=False)

    if args.backend == "torch":
        plddt_latent_config = omegaconf.OmegaConf.load(plddt_latent_config_path)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
        plddt_latent_module.load_state_dict(plddt_latent_checkpoint, strict=True)
        plddt_latent_module = plddt_latent_module.to(device)
    elif args.backend == "mlx":
        # replace torch implementations with mlx
        with open(plddt_latent_config_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        plddt_latent_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, plddt_latent_checkpoint.items()) if k is not None}
        plddt_latent_module.update(tree_unflatten(list(mlx_state_dict.items())))

    plddt_latent_module.eval()
    print(f"pLDDT latent module loaded with {args.backend} backend.")

    return plddt_latent_module, plddt_out_module


def initialize_esm_model(args, device):
    # load ESM2 model
    esm_model, esm_dict = load_local_or_hub_esm2_3b()
    af2_to_esm = _af2_to_esm(esm_dict)

    if args.backend == 'torch':
        esm_model = esm_model.to(device)
        af2_to_esm = af2_to_esm.to(device)
    elif args.backend == 'mlx':
        esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)
        esm_state_dict_torch = esm_model.cpu().state_dict()

        esm_state_dict_torch = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, esm_state_dict_torch.items()) if k is not None}
        esm_model_mlx.update(tree_unflatten(list(esm_state_dict_torch.items())))
        esm_model = esm_model_mlx
    print(f"pLM ESM-3B loaded with {args.backend} backend.")

    esm_model.eval()
    return esm_model, esm_dict, af2_to_esm


def initialize_others(args, device):
    # prepare data tokenizer, featurizer, and processor
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    processor = ProteinDataProcessor(
        device=device,
        scale=16.0, 
        ref_scale=5.0, 
        multiplicity=1,
        inference_multiplicity=args.nsample_per_protein,
        backend=args.backend,
    )

    # define flow process and sampler
    flow = LinearPath()

    if args.backend == "torch":
        sampler_cls = EMSampler
    elif args.backend == "mlx":
        sampler_cls = EMSamplerMLX

    sampler = sampler_cls(
        num_timesteps=args.num_steps,
        t_start=1e-4,
        tau=args.tau,
        log_timesteps=True,
        w_cutoff=0.99,
    )
    return tokenizer, featurizer, processor, flow, sampler


def generate_structure(
    args, batch, sampler, flow, processor,
    model, plddt_latent_module, plddt_out_module, device
):
    # run inference for target protein
    if args.backend == "torch":
        noise = torch.randn_like(batch['coords']).to(device)
    elif args.backend == "mlx":
        noise = mx.random.normal(batch['coords'].shape)
    out_dict = sampler.sample(model, flow, noise, batch)

    if args.plddt:
        if args.backend == "torch":
            t = torch.ones(batch['coords'].shape[0], device=device)
            # use unscaled coords to extract latent for pLDDT prediction
            out_feat = plddt_latent_module(
                out_dict["denoised_coords"].detach(), t, batch)
            plddt_out_dict = plddt_out_module(
                out_feat["latent"].detach(),
                batch,
            )
        elif args.backend == "mlx":
            t = mx.ones(batch['coords'].shape[0])
            # use unscaled coords to extract latent for pLDDT prediction
            out_feat = plddt_latent_module(
                out_dict["denoised_coords"], t, batch)
            plddt_out_dict = plddt_out_module(
                out_feat["latent"],
                batch,
            )
        # scale pLDDT to [0, 100]
        plddts = plddt_out_dict["plddt"] * 100.0
    else:
        plddts = None

    out_dict = processor.postprocess(out_dict, batch)
    # sampled_coord = out_dict['denoised_coords'].detach()
    if args.backend == "torch":
        sampled_coord = out_dict['denoised_coords'].detach()
    else:
        sampled_coord = out_dict['denoised_coords']

    pad_mask = batch['atom_pad_mask']
    return sampled_coord, pad_mask, plddts


def predict_structures_from_fastas(args):
    # create output directories
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / f"predictions_{args.simplefold_model}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    if args.backend == "mlx" and not MLX_AVAILABLE:
        args.backend = "torch"
        print("MLX not available, switch to torch backend.")

    # initialize models
    model, device = initialize_folding_model(args)
    plddt_latent_module, plddt_out_module = initialize_plddt_module(args, device)
    esm_model, esm_dict, af2_to_esm = initialize_esm_model(args, device)

    # initialize other components
    tokenizer, featurizer, processor, flow, sampler = initialize_others(args, device)

    # process fasta files to input format
    ccd_path = resolve_ccd_path(cache)
    data = check_fasta_inputs(_resolve_path(args.fasta_path))
    if not data:
        raise ValueError("No valid input files found. Please check the input directory.")
    process_fastas(
        data=data,
        out_dir=output_dir,
        ccd_path=ccd_path,
    )

    for struct_file in output_dir.glob("structures/*.npz"):
        record_file = output_dir / "records" / f"{struct_file.stem}.json"

        # prepare the target protein data for inference
        batch, structure, record = process_one_inference_structure(
            struct_file, record_file,
            tokenizer, featurizer, processor,
            esm_model, esm_dict, af2_to_esm,
        )

        sampled_coord, pad_mask, plddts = generate_structure(
            args, batch, sampler, flow, processor,
            model, plddt_latent_module, plddt_out_module, device
        )

        for i in range(args.nsample_per_protein):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]

            # save the generated structure
            structure_save = process_structure(
                deepcopy(structure), sampled_coord_i, pad_mask_i, record, backend=args.backend
            )
            outname = f"{record.id}_sampled_{i}"
            save_structure(
                structure_save, prediction_dir, outname,
                output_format=args.output_format,
                plddts=plddts[i] if plddts is not None else None
            )
