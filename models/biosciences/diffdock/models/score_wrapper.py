from argparse import Namespace
from functools import partial
from pathlib import Path

import torch
import yaml
from torch_geometric.nn.data_parallel import DataParallel

from onescience.utils.diffdock.diffusion_utils import get_timestep_embedding, t_to_sigma as t_to_sigma_compl
from onescience.utils.diffdock.utils import ExponentialMovingAverage

from .aa_model import AAModel
from .cg_model import CGModel
from .old_aa_model import AAOldModel


_LM_EMBEDDING_KEYS = (
    "moad_esm_embeddings_path",
    "pdbbind_esm_embeddings_path",
    "pdbsidechain_esm_embeddings_path",
    "esm_embeddings_path",
    "esm_embeddings_model",
)


def load_model_args(model_dir):
    model_dir = Path(model_dir)
    config_path = model_dir / "model_parameters.yml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.full_load(handle) or {}
    return Namespace(**config)


def model_uses_lm_embeddings(model_args):
    return any(getattr(model_args, key, None) is not None for key in _LM_EMBEDDING_KEYS)


def _has_arg(args, name):
    try:
        return name in args
    except TypeError:
        return hasattr(args, name)


def _get_arg(args, name, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    if _has_arg(args, name):
        return getattr(args, name)
    return default


def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False, old=False):
    timestep_emb_func = get_timestep_embedding(
        embedding_type=_get_arg(args, "embedding_type", "sinusoidal"),
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=_get_arg(args, "embedding_scale", 10000),
    )

    all_atoms = _get_arg(args, "all_atoms", False)
    if old and not all_atoms:
        raise NotImplementedError(
            "The old coarse-grained DiffDock model path is not migrated yet. "
            "Use old=True only with all_atoms=True, or migrate old_cg_model.py first."
        )

    lm_embedding_type = None
    if (
        _get_arg(args, "moad_esm_embeddings_path") is not None
        or _get_arg(args, "pdbbind_esm_embeddings_path") is not None
        or _get_arg(args, "pdbsidechain_esm_embeddings_path") is not None
        or _get_arg(args, "esm_embeddings_path") is not None
    ):
        lm_embedding_type = "precomputed"
    if _get_arg(args, "esm_embeddings_model") is not None:
        lm_embedding_type = args.esm_embeddings_model

    if old:
        model_class = AAOldModel
    elif all_atoms:
        model_class = AAModel
    else:
        model_class = CGModel

    model_kwargs = dict(
        t_to_sigma=t_to_sigma,
        device=device,
        no_torsion=args.no_torsion,
        timestep_emb_func=timestep_emb_func,
        num_conv_layers=args.num_conv_layers,
        lig_max_radius=args.max_radius,
        scale_by_sigma=args.scale_by_sigma,
        sigma_embed_dim=args.sigma_embed_dim,
        norm_by_sigma=_get_arg(args, "norm_by_sigma", False),
        ns=args.ns,
        nv=args.nv,
        distance_embed_dim=args.distance_embed_dim,
        cross_distance_embed_dim=args.cross_distance_embed_dim,
        batch_norm=not args.no_batch_norm,
        dropout=args.dropout,
        use_second_order_repr=args.use_second_order_repr,
        cross_max_distance=args.cross_max_distance,
        dynamic_max_cross=args.dynamic_max_cross,
        smooth_edges=_get_arg(args, "smooth_edges", False),
        odd_parity=_get_arg(args, "odd_parity", False),
        lm_embedding_type=lm_embedding_type,
        confidence_mode=confidence_mode,
        confidence_dropout=_get_arg(args, "confidence_dropout", 0.0),
        confidence_no_batchnorm=_get_arg(args, "confidence_no_batchnorm", False),
        affinity_prediction=_get_arg(args, "affinity_prediction", False),
        parallel=_get_arg(args, "parallel", 1),
        num_confidence_outputs=(
            len(args.rmsd_classification_cutoff) + 1
            if isinstance(_get_arg(args, "rmsd_classification_cutoff"), list)
            else 1
        ),
        atom_num_confidence_outputs=(
            len(args.atom_rmsd_classification_cutoff) + 1
            if isinstance(_get_arg(args, "atom_rmsd_classification_cutoff"), list)
            else 1
        ),
        parallel_aggregators=_get_arg(args, "parallel_aggregators", ""),
        fixed_center_conv=not _get_arg(args, "not_fixed_center_conv", False),
        no_aminoacid_identities=_get_arg(args, "no_aminoacid_identities", False),
        include_miscellaneous_atoms=_get_arg(args, "include_miscellaneous_atoms", False),
        sh_lmax=_get_arg(args, "sh_lmax", 2),
        differentiate_convolutions=not _get_arg(args, "no_differentiate_convolutions", False),
        tp_weights_layers=_get_arg(args, "tp_weights_layers", 2),
        num_prot_emb_layers=_get_arg(args, "num_prot_emb_layers", 0),
        reduce_pseudoscalars=_get_arg(args, "reduce_pseudoscalars", False),
        embed_also_ligand=_get_arg(args, "embed_also_ligand", False),
        atom_confidence=_get_arg(args, "atom_confidence_loss_weight", 0.0) > 0.0,
        sidechain_pred=(
            (_has_arg(args, "sidechain_loss_weight") and args.sidechain_loss_weight > 0)
            or (_has_arg(args, "backbone_loss_weight") and args.backbone_loss_weight > 0)
        ),
        depthwise_convolution=_get_arg(args, "depthwise_convolution", False),
    )
    if model_class is AAModel:
        model_kwargs["crop_beyond"] = _get_arg(args, "crop_beyond", None)
    elif model_class is AAOldModel:
        for key in (
            "atom_num_confidence_outputs",
            "differentiate_convolutions",
            "tp_weights_layers",
            "num_prot_emb_layers",
            "reduce_pseudoscalars",
            "embed_also_ligand",
            "atom_confidence",
            "sidechain_pred",
            "depthwise_convolution",
        ):
            model_kwargs.pop(key, None)
        model_kwargs["lm_embedding_type"] = (
            "esm" if _get_arg(args, "esm_embeddings_path") is not None else None
        )
        model_kwargs["use_old_atom_encoder"] = _get_arg(args, "use_old_atom_encoder", True)

    model = model_class(**model_kwargs)

    if device.type == "cuda" and not no_parallel and _get_arg(args, "dataset") != "torsional":
        model = DataParallel(model)
    model.to(device)
    return model


def build_score_model(
    model_args,
    device,
    no_parallel=False,
    confidence_mode=False,
    old=False,
):
    t_to_sigma = partial(t_to_sigma_compl, args=model_args)
    model = get_model(
        model_args,
        device,
        t_to_sigma=t_to_sigma,
        no_parallel=no_parallel,
        confidence_mode=confidence_mode,
        old=old,
    )
    return model, t_to_sigma


def load_score_model(
    model_dir,
    ckpt,
    device=None,
    no_parallel=True,
    confidence_mode=False,
    old=False,
    strict=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = load_model_args(model_dir)
    model, t_to_sigma = build_score_model(
        model_args=model_args,
        device=device,
        no_parallel=no_parallel,
        confidence_mode=confidence_mode,
        old=old,
    )

    checkpoint_path = Path(model_dir) / ckpt
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if isinstance(state_dict, dict) and "model" in state_dict and "optimizer" in state_dict:
        model.load_state_dict(state_dict["model"], strict=strict)
        if "ema_weights" in state_dict and getattr(model_args, "ema_rate", None) is not None:
            ema_weights = ExponentialMovingAverage(model.parameters(), decay=model_args.ema_rate)
            ema_weights.load_state_dict(state_dict["ema_weights"], device=device)
            ema_weights.copy_to(model.parameters())
    else:
        model.load_state_dict(state_dict, strict=strict)
    model = model.to(device)
    model.eval()
    return model, model_args, t_to_sigma
