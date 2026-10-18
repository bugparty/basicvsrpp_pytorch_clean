# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Checkpoint loading without mmengine."""

from __future__ import annotations

import os
import pickle
import re
import types
from pathlib import Path
from typing import Any, Dict, Union
from urllib.parse import urlparse

import torch


class _MissingObject:
    """Stand-in for objects whose class cannot be imported (e.g. mmengine's message hub)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        self.__dict__['_state'] = state


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (_MissingObject,), {'__module__': module})


_tolerant_pickle = types.ModuleType('_tolerant_pickle')
_tolerant_pickle.Unpickler = _TolerantUnpickler
_tolerant_pickle.load = lambda f, **kwargs: _TolerantUnpickler(f, **kwargs).load()


def _resolve(path: Union[str, Path]) -> str:
    path = str(path)
    if not path.startswith(('http://', 'https://')):
        return path
    cache_dir = os.path.join(torch.hub.get_dir(), 'checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, os.path.basename(urlparse(path).path))
    if not os.path.exists(cached):
        torch.hub.download_url_to_file(path, cached)
    return cached


def load_checkpoint_file(path: Union[str, Path], map_location: Any = 'cpu') -> Any:
    """Load a checkpoint file or URL, safely when it only holds tensors.

    Full training checkpoints also pickle optimizer/message-hub objects. Those are
    loaded with the regular (unsafe) unpickler, as mmengine did, so only use
    checkpoints from trusted sources; classes that are not installed are replaced
    by placeholders because only the weights are needed.
    """
    path = _resolve(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        return torch.load(path, map_location=map_location, weights_only=False,
                          pickle_module=_tolerant_pickle)


def load_state_dict(path: Union[str, Path], map_location: Any = 'cpu') -> Dict[str, torch.Tensor]:
    """Return the model state dict of a checkpoint, unwrapping ``state_dict`` and DDP ``module.`` prefixes."""
    checkpoint = load_checkpoint_file(path, map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    return {re.sub(r'^module\.', '', k): v for k, v in checkpoint.items()}
