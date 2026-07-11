"""Compatibility helpers for building Megatron indexed datasets.

The legacy JSON preprocessing script expects ``indexed_dataset.make_builder``.
In the standalone package we keep that small API surface and delegate the
actual dataset writing to Megatron Core.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _dtype_from_vocab_size(vocab_size: int | None) -> type[np.number]:
    """Choose a compact dtype that can hold token ids for the vocabulary."""
    if vocab_size is None:
        return np.int32
    if vocab_size <= np.iinfo(np.uint8).max + 1:
        return np.uint8
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.uint16
    return np.int32


class _IndexedDatasetBuilderAdapter:
    """Adapter matching the small builder API used by preprocess_data_json."""

    def __init__(self, bin_path: str, dtype: type[np.number]) -> None:
        from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

        self.dtype = dtype
        self._builder = IndexedDatasetBuilder(bin_path=bin_path, dtype=dtype)

    def add_item(self, item: Any) -> None:
        import torch

        if isinstance(item, torch.Tensor):
            tensor = item
        else:
            tensor = torch.as_tensor(np.asarray(item, dtype=self.dtype))
        self._builder.add_item(tensor)

    def end_document(self) -> None:
        self._builder.end_document()

    def finalize(self, idx_path: str) -> None:
        self._builder.finalize(idx_path)


def make_builder(
    out_file: str,
    impl: str = "mmap",
    vocab_size: int | None = None,
    dtype: type[np.number] | None = None,
) -> _IndexedDatasetBuilderAdapter:
    """Create a Megatron indexed dataset builder.

    Args:
        out_file: Output ``.bin`` file path.
        impl: Dataset implementation. Megatron Core writes mmap datasets here.
        vocab_size: Optional vocabulary size used to infer storage dtype.
        dtype: Optional explicit NumPy dtype.
    """
    if impl != "mmap":
        raise ValueError(
            "Only the 'mmap' dataset implementation is supported by this "
            "standalone Evo2 compatibility layer."
        )
    return _IndexedDatasetBuilderAdapter(
        bin_path=out_file,
        dtype=dtype or _dtype_from_vocab_size(vocab_size),
    )
