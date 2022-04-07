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

from typing import Callable, Optional

import torch

from monai.data.meta_obj import MetaObj, get_track_meta, get_track_transforms

__all__ = ["MetaTensor"]


class MetaTensor(MetaObj, torch.Tensor):
    """
    Class that inherits from both `torch.Tensor` and `MetaObj`, adding support for meta
    data.

    Meta data is stored in the form of of a dictionary. Affine matrices are stored in
    the form of `torch.Tensor`.

    We store the affine as its own element, so that this can be updated by
    transforms. All other meta data that we don't plan on touching until we
    need to save the image to file lives in `meta`.

    Behavior should be the same as `torch.Tensor` aside from the extended
    meta functionality.

    Copying of information:
        * For `c = a + b`, then auxiliary data (e.g., meta data) will be copied from the
        first instance of `MetaTensor`.

    Example:
        .. code-block:: python

            import torch
            from monai.data import MetaTensor

            t = torch.tensor([1,2,3])
            meta = {"some": "info"}
            affine = torch.eye(4)
            m = MetaTensor(t, meta=meta, affine=affine)
            m2 = m+m
            assert isinstance(m2, MetaTensor)
            assert m2.meta == meta
    """

    @staticmethod
    def __new__(cls, x, affine: torch.Tensor | None = None, meta: dict | None = None, *args, **kwargs) -> MetaTensor:
        return torch.as_tensor(x, *args, **kwargs).as_subclass(cls)  # type: ignore

    def __init__(self, x, affine: torch.Tensor | None = None, meta: dict | None = None) -> None:
        """If `affine` is given, use it. Else, if `affine` exists in the input tensor, use it. Else, use
        the default value. The same is true for `meta`."""
        self._set_initial_val("affine", affine, x, self.get_default_affine)
        self._set_initial_val("meta", meta, x, self.get_default_meta)

    def _copy_attr(
        self, attribute: str, input_objs: list[MetaObj], default_fn: Callable, deep_copy: bool
    ) -> None:
        super()._copy_attr(attribute, input_objs, default_fn, deep_copy)
        val = getattr(self, attribute)
        if isinstance(self, torch.Tensor) and isinstance(val, torch.Tensor):
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
        return ret

    def get_default_affine(self) -> torch.Tensor:
        return torch.eye(4, device=self.device)

    def as_tensor(self) -> torch.Tensor:
        """
        Return the `MetaTensor` as a `torch.Tensor`.
        It is OS dependent as to whether this will be a deep copy or not.
        """
        return self.as_subclass(torch.Tensor)  # type: ignore
