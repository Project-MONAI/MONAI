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

import numpy as np
import torch

_TRACK_META = True
_TRACK_TRANSFORMS = True

__all__ = ["get_track_meta", "get_track_transforms", "set_track_meta", "set_track_transforms", "MetaObj"]


def set_track_meta(val: bool) -> None:
    """
    Boolean to set whether metadata is tracked. If `True`,
    `MetaTensor` will be returned where appropriate. If `False`,
    `torch.Tensor` will be returned instead.
    """
    global _TRACK_META
    _TRACK_META = val


def set_track_transforms(val: bool) -> None:
    """
    Boolean to set whether transforms are tracked.
    """
    global _TRACK_TRANSFORMS
    _TRACK_TRANSFORMS = val


def get_track_meta() -> bool:
    """
    Get track meta data boolean.
    """
    global _TRACK_META
    return _TRACK_META


def get_track_transforms() -> bool:
    """
    Get track transform boolean.
    """
    global _TRACK_TRANSFORMS
    return _TRACK_TRANSFORMS


class MetaObj:
    """
    Class that stores meta and affine.

    We store the affine as its own element, so that this can be updated by
    transforms. All other meta data that we don't plan on touching until we
    need to save the image to file lives in `meta`.

    This allows for subclassing `np.ndarray` and `torch.Tensor`.

    Copying metadata:
        * For `c = a + b`, then the meta data will be copied from the first
        instance of `MetaImage`.
    """

    _meta: dict
    _affine: torch.Tensor

    def set_initial_val(self, attribute: str, input_arg: Any, input_tensor: MetaObj, default_fn: Callable) -> None:
        """
        Set the initial value. Try to use input argument, but if this is None
        and there is a MetaImage input, then copy that. Failing both these two,
        use a default value.
        """
        if input_arg is None:
            input_arg = getattr(input_tensor, attribute, None)
        if input_arg is None:
            input_arg = default_fn(self)
        setattr(self, attribute, input_arg)

    @staticmethod
    def get_tensors_or_arrays(args: Sequence[Any]) -> list[MetaObj]:
        """
        Recursively extract all instances of `MetaObj`.
        Works for `torch.add(a, b)`, `torch.stack([a, b])` and numpy equivalents.
        """
        out = []
        for a in args:
            if isinstance(a, (list, tuple)):
                out += MetaObj.get_tensors_or_arrays(a)
            elif isinstance(a, MetaObj):
                out.append(a)
        return out

    def _copy_attr(
        self, attribute: str, input_objs: list[MetaObj], default_fn: Callable, deepcopy_required: bool
    ) -> None:
        """
        Copy an attribute from the first in a list of `MetaObj`
        In the cases `torch.add(a, b)` and `torch.add(input=a, other=b)`,
        both `a` and `b` could be `MetaObj` or `torch.Tensor` so check
        them all. Copy the first to the output, and make sure on correct
        device.
        Might have the MetaObj nested in list, e.g., `torch.stack([a, b])`.
        """
        attributes = [getattr(i, attribute) for i in input_objs]
        if len(attributes) > 0:
            val = attributes[0]
            if deepcopy_required:
                val = deepcopy(val)
            if isinstance(self, torch.Tensor) and isinstance(val, torch.Tensor):
                val = val.to(self.device)
            setattr(self, attribute, val)
        else:
            setattr(self, attribute, default_fn())

    def _copy_meta(self, input_meta_objs: list[MetaObj]) -> None:
        """
        Copy meta data from a list of `MetaObj`.
        If there has been a change in `id` (e.g., `a+b`), then deepcopy. Else (e.g., `a+=1`), don't.
        """
        id_in = id(input_meta_objs[0]) if len(input_meta_objs) > 0 else None
        deepcopy_required = id(self) != id_in
        attributes = ("affine", "meta")
        default_fns: tuple[Callable, ...] = (self.get_default_affine, self.get_default_meta)
        for attribute, default_fn in zip(attributes, default_fns):
            self._copy_attr(attribute, input_meta_objs, default_fn, deepcopy_required)

    def get_default_meta(self) -> dict:
        return {}

    def get_default_affine(self) -> torch.Tensor | np.ndarray:
        raise NotImplementedError()

    def __repr__(self) -> str:
        """String representation of class."""
        out: str = super().__repr__()

        out += f"\nAffine\n{self.affine}"

        out += "\nMetaData\n"
        if self.meta is not None:
            out += "".join(f"\t{k}: {v}\n" for k, v in self.meta.items())
        else:
            out += "None"

        return out

    @property
    def affine(self) -> torch.Tensor:
        return self._affine

    @affine.setter
    def affine(self, d: torch.Tensor) -> None:
        self._affine = d

    @property
    def meta(self) -> dict:
        return self._meta

    @meta.setter
    def meta(self, d: dict):
        self._meta = d
