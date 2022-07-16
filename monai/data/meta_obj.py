# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import itertools
import pprint
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import torch

from monai.utils.enums import TraceKeys
from monai.utils.misc import first

_TRACK_META = True

__all__ = ["get_track_meta", "set_track_meta", "MetaObj"]


def set_track_meta(val: bool) -> None:
    """
    Boolean to set whether metadata is tracked. If `True`, metadata will be associated
    its data by using subclasses of `MetaObj`. If `False`, then data will be returned
    with empty metadata.

    If `set_track_meta` is `False`, then standard data objects will be returned (e.g.,
    `torch.Tensor` and `np.ndarray`) as opposed to MONAI's enhanced objects.

    By default, this is `True`, and most users will want to leave it this way. However,
    if you are experiencing any problems regarding metadata, and aren't interested in
    preserving metadata, then you can disable it.
    """
    global _TRACK_META
    _TRACK_META = val


def get_track_meta() -> bool:
    """
    Return the boolean as to whether metadata is tracked. If `True`, metadata will be
    associated its data by using subclasses of `MetaObj`. If `False`, then data will be
    returned with empty metadata.

    If `set_track_meta` is `False`, then standard data objects will be returned (e.g.,
    `torch.Tensor` and `np.ndarray`) as opposed to MONAI's enhanced objects.

    By default, this is `True`, and most users will want to leave it this way. However,
    if you are experiencing any problems regarding metadata, and aren't interested in
    preserving metadata, then you can disable it.
    """
    return _TRACK_META


class MetaObj:
    """
    Abstract base class that stores data as well as any extra metadata.

    This allows for subclassing `torch.Tensor` and `np.ndarray` through multiple inheritance.

    Metadata is stored in the form of a dictionary.

    Behavior should be the same as extended class (e.g., `torch.Tensor` or `np.ndarray`)
    aside from the extended meta functionality.

    Copying of information:

        * For `c = a + b`, then auxiliary data (e.g., metadata) will be copied from the
          first instance of `MetaObj` if `a.is_batch` is False
          (For batched data, the metadata will be shallow copied for efficiency purposes).

    """

    def __init__(self):
        self._meta: dict = MetaObj.get_default_meta()
        self._applied_operations: list = MetaObj.get_default_applied_operations()
        self._is_batch: bool = False

    @staticmethod
    def flatten_meta_objs(*args: Iterable):
        """
        Recursively flatten input and yield all instances of `MetaObj`.
        This means that for both `torch.add(a, b)`, `torch.stack([a, b])` (and
        their numpy equivalents), we return `[a, b]` if both `a` and `b` are of type
        `MetaObj`.

        Args:
            args: Iterables of inputs to be flattened.
        Returns:
            list of nested `MetaObj` from input.
        """
        for a in itertools.chain(*args):
            if isinstance(a, (list, tuple)):
                yield from MetaObj.flatten_meta_objs(a)
            elif isinstance(a, MetaObj):
                yield a

    def _copy_attr(self, attributes: list[str], input_objs) -> None:
        """Copy attributes from the first in a list of `MetaObj`."""
        f = first(input_objs).__dict__
        self.__dict__.update({a: MetaObj.copy_items(f[a]) for a in attributes if a in f})

    @staticmethod
    def copy_items(data):
        """deepcopying items"""
        if isinstance(data, (list, dict, np.ndarray)):
            return data.copy()
        if isinstance(data, torch.Tensor):
            return data.detach().clone()
        return deepcopy(data)

    def _copy_meta(self, input_objs, deep_copy=False) -> None:
        """
        Copy metadata from an iterable of `MetaObj` instances. For a given attribute, we copy the
        adjunct data from the first element in the list containing that attribute.

        Args:
            input_objs: list of `MetaObj` to copy data from.

        """
        if not deep_copy:
            self.__dict__ = dict(first(input_objs).__dict__)  # shallow copy for performance
            return
        self._copy_attr(["_meta", "_applied_operations"], input_objs)

    @staticmethod
    def get_default_meta() -> dict:
        """Get the default meta.

        Returns:
            default metadata.
        """
        return {}

    @staticmethod
    def get_default_applied_operations() -> list:
        """Get the default applied operations.

        Returns:
            default applied operations.
        """
        return []

    def __repr__(self) -> str:
        """String representation of class."""
        out: str = "\nMetaData\n"
        if self.meta is not None:
            out += "".join(f"\t{k}: {v}\n" for k, v in self.meta.items())
        else:
            out += "None"

        out += "\nApplied operations\n"
        if self.applied_operations is not None:
            out += pprint.pformat(self.applied_operations, indent=2, compact=True, width=120)
        else:
            out += "None"

        out += f"\nIs batch?: {self.is_batch}"

        return out

    @property
    def meta(self) -> dict:
        """Get the meta. Defaults to ``{}``."""
        return self._meta if hasattr(self, "_meta") else MetaObj.get_default_meta()

    @meta.setter
    def meta(self, d) -> None:
        """Set the meta."""
        if d == TraceKeys.NONE:
            self._meta = MetaObj.get_default_meta()
        self._meta = d

    @property
    def applied_operations(self) -> list[dict]:
        """Get the applied operations. Defaults to ``[]``."""
        if hasattr(self, "_applied_operations"):
            return self._applied_operations
        return MetaObj.get_default_applied_operations()

    @applied_operations.setter
    def applied_operations(self, t) -> None:
        """Set the applied operations."""
        if t == TraceKeys.NONE:
            # received no operations when decollating a batch
            self._applied_operations = MetaObj.get_default_applied_operations()
            return
        self._applied_operations = t

    def push_applied_operation(self, t: Any) -> None:
        self._applied_operations.append(t)

    def pop_applied_operation(self) -> Any:
        return self._applied_operations.pop()

    @property
    def is_batch(self) -> bool:
        """Return whether object is part of batch or not."""
        return self._is_batch if hasattr(self, "_is_batch") else False

    @is_batch.setter
    def is_batch(self, val: bool) -> None:
        """Set whether object is part of batch or not."""
        self._is_batch = val
