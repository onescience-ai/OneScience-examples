from contextlib import contextmanager
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint


class _CheckpointWrapper(torch.nn.Module):
    def __init__(self, checkpoint_fn, module):
        super().__init__()
        self.checkpoint_fn = checkpoint_fn
        self.module = module

    def __call__(self, *args, **kwargs):
        checkpointed = partial(self.checkpoint_fn, self.module, use_reentrant=True)
        return checkpointed(*args, **kwargs)


@contextmanager
def replace_function(module, replace_layers_list, ddp_flag=False):
    def _get_by_path(root, path):
        for key in path.split("."):
            root = root[int(key)] if key.isdigit() else getattr(root, key)
        return root

    def _set_by_path(root, path, value):
        keys = path.split(".")
        for key in keys[:-1]:
            root = root[int(key)] if key.isdigit() else getattr(root, key)
        last = keys[-1]
        if last.isdigit():
            root[int(last)] = value
        else:
            setattr(root, last, value)

    base = module.module if ddp_flag else module
    stash = []
    for path in replace_layers_list:
        original = _get_by_path(base, path)
        stash.append((path, original))
        _set_by_path(base, path, _CheckpointWrapper(checkpoint, original))

    try:
        yield
    finally:
        for path, original in stash:
            _set_by_path(base, path, original)
