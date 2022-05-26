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

from copy import deepcopy
from typing import Any, Callable, Sequence

_TRACK_META = True

__all__ = ["get_track_meta", "set_track_meta", "MetaObj"]


def set_track_meta(val: bool) -> None:
    """
    Boolean to set whether metadata is tracked. If `True`, metadata will be associated
    its data by using subclasses of `MetaObj`. If `False`, then data will be returned
    with empty metadata.

    If `set_track_meta` is `False`, then standard data objects will be returned (e.g.,
    `torch.Tensor` and `np.ndarray`) as opposed to our enhanced objects.

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
    `torch.Tensor` and `np.ndarray`) as opposed to our enhanced objects.

    By default, this is `True`, and most users will want to leave it this way. However,
    if you are experiencing any problems regarding metadata, and aren't interested in
    preserving metadata, then you can disable it.
    """
    return _TRACK_META


class MetaObj:
    """
    Abstract base class that stores data as well as any extra metadata.

    This allows for subclassing `torch.Tensor` and `np.ndarray` through multiple
    inheritance.

    Metadata is stored in the form of a dictionary.

    Behavior should be the same as extended class (e.g., `torch.Tensor` or `np.ndarray`)
    aside from the extended meta functionality.

    Copying of information:

        * For `c = a + b`, then auxiliary data (e.g., metadata) will be copied from the
          first instance of `MetaObj`.

    """

    def __init__(self):
        self._meta: dict = self.get_default_meta()
        self._is_batch: bool = False

    @staticmethod
    def flatten_meta_objs(args: Sequence[Any]) -> list[MetaObj]:
        """
        Recursively flatten input and return all instances of `MetaObj` as a single
        list. This means that for both `torch.add(a, b)`, `torch.stack([a, b])` (and
        their numpy equivalents), we return `[a, b]` if both `a` and `b` are of type
        `MetaObj`.

        Args:
            args: Sequence of inputs to be flattened.
        Returns:
            list of nested `MetaObj` from input.
        """
        out = []
        for a in args:
            if isinstance(a, (list, tuple)):
                out += MetaObj.flatten_meta_objs(a)
            elif isinstance(a, MetaObj):
                out.append(a)
        return out

    def _copy_attr(self, attribute: str, input_objs: list[MetaObj], default_fn: Callable, deep_copy: bool) -> None:
        """
        Copy an attribute from the first in a list of `MetaObj`. In the case of
        `torch.add(a, b)`, both `a` and `b` could be `MetaObj` or something else, so
        check them all. Copy the first to `self`.

        We also perform a deep copy of the data if desired.

        Args:
            attribute: string corresponding to attribute to be copied (e.g., `meta`).
            input_objs: List of `MetaObj`. We'll copy the attribute from the first one
                that contains that particular attribute.
            default_fn: If none of `input_objs` have the attribute that we're
                interested in, then use this default function (e.g., `lambda: {}`.)
            deep_copy: Should the attribute be deep copied? See `_copy_meta`.

        Returns:
            Returns `None`, but `self` should be updated to have the copied attribute.
        """
        attributes = [getattr(i, attribute) for i in input_objs if hasattr(i, attribute)]
        if len(attributes) > 0:
            val = attributes[0]
            if deep_copy:
                val = deepcopy(val)
            setattr(self, attribute, val)
        else:
            setattr(self, attribute, default_fn())

    def _copy_meta(self, input_objs: list[MetaObj]) -> None:
        """
        Copy metadata from a list of `MetaObj`. For a given attribute, we copy the
        adjunct data from the first element in the list containing that attribute.

        If there has been a change in `id` (e.g., `a=b+c`), then deepcopy. Else (e.g.,
        `a+=1`), then don't.

        Args:
            input_objs: list of `MetaObj` to copy data from.

        """
        id_in = id(input_objs[0]) if len(input_objs) > 0 else None
        deep_copy = id(self) != id_in
        self._copy_attr("meta", input_objs, self.get_default_meta, deep_copy)
        self._copy_attr("applied_operations", input_objs, self.get_default_applied_operations, deep_copy)
        self.is_batch = input_objs[0].is_batch if len(input_objs) > 0 else False

    def get_default_meta(self) -> dict:
        """Get the default meta.

        Returns:
            default metadata.
        """
        return {}

    def get_default_applied_operations(self) -> list:
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
            for i in self.applied_operations:
                out += f"\t{str(i)}\n"
        else:
            out += "None"

        out += f"\nIs batch?: {self.is_batch}"

        return out

    @property
    def meta(self) -> dict:
        """Get the meta."""
        return self._meta

    @meta.setter
    def meta(self, d: dict) -> None:
        """Set the meta."""
        self._meta = d

    @property
    def applied_operations(self) -> list:
        """Get the applied operations."""
        return self._applied_operations

    @applied_operations.setter
    def applied_operations(self, t: list) -> None:
        """Set the applied operations."""
        self._applied_operations = t

    def push_applied_operation(self, t: Any) -> None:
        self._applied_operations.append(t)

    def pop_applied_operation(self) -> Any:
        return self._applied_operations.pop()

    @property
    def is_batch(self) -> bool:
        """Return whether object is part of batch or not."""
        return self._is_batch

    @is_batch.setter
    def is_batch(self, val: bool) -> None:
        """Set whether object is part of batch or not."""
        self._is_batch = val
