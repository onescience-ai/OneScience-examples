"""Standalone compact MetNet-3 adaptation."""

from .metnet3 import MetNet3, MetNet3Config
from .metnet3_losses import multitask_loss
from .metnet3_schema import validate_batch

__all__ = ["MetNet3", "MetNet3Config", "multitask_loss", "validate_batch"]
