"""Self-contained utility helpers used by the scGPT model package.

Extracted from ``onescience.utils.scgpt.util`` so that the ``model/``
directory does not depend on the installed ``onescience`` package.
"""

import re
from typing import List, Mapping, Optional, Union

import numpy as np
import torch

from .logging import logger


def map_raw_id_to_vocab_id(
    raw_ids: Union[np.ndarray, torch.Tensor],
    gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the

    Args:
        raw_ids: the raw ids to map
        gene_ids: the gene ids to map to
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError(f"raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)


def load_pretrained(
    model: torch.nn.Module,
    pretrained_params: Mapping[str, torch.Tensor],
    strict: bool = False,
    prefix: Optional[List[str]] = None,
    verbose: bool = True,
) -> torch.nn.Module:
    """
    Load pretrained weights to the model.

    Args:
        model (torch.nn.Module): The model to load weights to.
        pretrained_params (Mapping[str, torch.Tensor]): The pretrained parameters.
        strict (bool): Whether to strictly enforce that the keys in :attr:`pretrained_params`
            match the keys returned by this module's :meth:`Module.state_dict`. Default to False.
        prefix (List[str]): The list of prefix strings to match with the keys in
            :attr:`pretrained_params`. The matched keys will be loaded. Default to None.

    Returns:
        torch.nn.Module: The model with pretrained weights.
    """

    pretrained_params = dict(pretrained_params)

    use_flash_attn = getattr(model, "use_fast_transformer", True)
    if not use_flash_attn:
        rename_rules = {
            r"self_attn\._impl\.Wqkv\.": "self_attn.in_proj_",
            r"self_attn\.Wqkv\.": "self_attn.in_proj_",
            r"self_attn\._impl\.out_proj\.": "self_attn.out_proj.",
        }
        pretrained_params = {
            _rename_key(k, rename_rules): v for k, v in pretrained_params.items()
        }
    else:
        # Import locally to avoid a model <-> utils import cycle at module import time.
        from .flash_attn_compat import (
            get_flash_attn_parameter_rename_rules,
        )

        rename_rules = get_flash_attn_parameter_rename_rules(pretrained_params)
        if rename_rules:
            pretrained_params = {
                _rename_key(k, rename_rules): v for k, v in pretrained_params.items()
            }

    if prefix is not None and len(prefix) > 0:
        if isinstance(prefix, str):
            prefix = [prefix]
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if any(k.startswith(p) for p in prefix)
        }

    model_dict = model.state_dict()
    if strict:
        if verbose:
            for k, v in pretrained_params.items():
                logger.info(f"Loading parameter {k} with shape {v.shape}")
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
    else:
        if verbose:
            for k, v in pretrained_params.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    logger.info(f"Loading parameter {k} with shape {v.shape}")
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)

    return model


def _rename_key(key: str, rename_rules: Mapping[str, str]) -> str:
    for pattern, replacement in rename_rules.items():
        key = re.sub(pattern, replacement, key)
    return key
