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

import warnings
from typing import Callable

import torch

from monai.data.meta_obj import MetaObj, get_track_meta, get_track_transforms
from monai.utils.enums import PostFix

__all__ = ["MetaTensor"]


class MetaTensor(MetaObj, torch.Tensor):
    """
    Class that inherits from both `torch.Tensor` and `MetaObj`, adding support for meta
    data.

    Metadata is stored in the form of a dictionary. Nested, an affine matrix will be
    stored. This should be in the form of `torch.Tensor`.

    Behavior should be the same as `torch.Tensor` aside from the extended
    meta functionality.

    Copying of information:
        * For `c = a + b`, then auxiliary data (e.g., metadata) will be copied from the
        first instance of `MetaTensor`.

    Example:
        .. code-block:: python

            import torch
            from monai.data import MetaTensor

            t = torch.tensor([1,2,3])
            affine = torch.eye(4) * 100
            meta = {"some": "info"}
            m = MetaTensor(t, affine=affine, meta=meta)
            m2 = m+m
            assert isinstance(m2, MetaTensor)
            assert m2.meta["some"] == "info"
            assert m2.affine = affine

    Notes:
        - Older versions of pytorch (<=1.8), `torch.jit.trace(net, im)` may
            not work if `im` is of type `MetaTensor`. This can be resolved with
            `torch.jit.trace(net, im.as_tensor())`.
        - A warning will be raised if in the constructor `affine` is not `None` and
            `meta` already contains the key `affine`.
    """

    @staticmethod
    def __new__(cls, x, affine: torch.Tensor | None = None, meta: dict | None = None, *args, **kwargs) -> MetaTensor:
        """
        If `meta` is given, use it. Else, if `meta` exists in the input tensor, use it.
        Else, use the default value. Similar for the affin, except this could come from
        four places.
        Priority: `affine`, `meta["affine"]`, `x.affine`, `get_default_affine`.
        """
        out: MetaTensor = torch.as_tensor(x, *args, **kwargs).as_subclass(cls)  # type: ignore
        # set meta
        if meta is not None:
            out.meta = meta
        elif isinstance(x, MetaObj):
            out.meta = x.meta
        else:
            out.meta = out.get_default_meta()
        # set the affine
        if affine is not None:
            if "affine" in out.meta:
                warnings.warn("Setting affine, but the applied meta contains an affine. " "This will be overwritten.")
            out.affine = affine
        elif "affine" in out.meta:
            pass  # nothing to do
        elif isinstance(x, MetaTensor):
            out.affine = x.affine
        else:
            out.affine = out.get_default_affine()
        out.affine = out.affine.to(out.device)

        return out

    def _copy_attr(self, attribute: str, input_objs: list[MetaObj], default_fn: Callable, deep_copy: bool) -> None:
        super()._copy_attr(attribute, input_objs, default_fn, deep_copy)
        val = getattr(self, attribute)
        if isinstance(val, torch.Tensor):
            setattr(self, attribute, val.to(self.device))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None) -> torch.Tensor:
        """Wraps all torch functions."""
        if kwargs is None:
            kwargs = {}
        ret: MetaTensor = super().__torch_function__(func, types, args, kwargs)
        # e.g., __repr__ returns a string
        if not isinstance(ret, torch.Tensor):
            return ret
        if not (get_track_meta() or get_track_transforms()):
            return ret.as_tensor()
        meta_args = MetaObj.flatten_meta_objs(list(args) + list(kwargs.values()))
        ret._copy_meta(meta_args)
        ret.affine = ret.affine.to(ret.device)
        return ret

    def get_default_affine(self) -> torch.Tensor:
        return torch.eye(4, device=self.device)

    def as_tensor(self) -> torch.Tensor:
        """
        Return the `MetaTensor` as a `torch.Tensor`.
        It is OS dependent as to whether this will be a deep copy or not.
        """
        return self.as_subclass(torch.Tensor)  # type: ignore

    def as_dict(self, key: str) -> dict:
        """
        Get the object as a dictionary for backwards compatibility.

        Args:
            key: Base key to store main data. The key for the metadata will be
                determined using `PostFix.meta`.

        Return:
            A dictionary consisting of two keys, the main data (stored under `key`) and
                the metadata.
        """
        return {key: self.as_tensor(), PostFix.meta(key): self.meta}

    @property
    def affine(self) -> torch.Tensor:
        """Get the affine."""
        return self.meta["affine"]  # type: ignore

    @affine.setter
    def affine(self, d: torch.Tensor) -> None:
        """Set the affine."""
        self.meta["affine"] = d
