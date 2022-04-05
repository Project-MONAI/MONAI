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

import torch

from monai.data.meta_obj import MetaObj, get_track_meta, get_track_transforms

__all__ = ["MetaTensor"]


class MetaTensor(MetaObj, torch.Tensor):
    """
    Class that extends upon `torch.Tensor`, adding support for meta data.

    We store the affine as its own element, so that this can be updated by
    transforms. All other meta data that we don't plan on touching until we
    need to save the image to file lives in `meta`.

    Behavior should be the same as `torch.Tensor` aside from the extended
    functionality.

    Copying metadata:
        * For `c = a + b`, then the meta data will be copied from the first
        instance of `MetaTensor`.
    """

    @staticmethod
    def __new__(cls, x, affine: torch.Tensor | None = None, meta: dict | None = None, *args, **kwargs) -> MetaTensor:
        return torch.as_tensor(x, *args, **kwargs).as_subclass(cls)  # type: ignore

    def __init__(self, x, affine: torch.Tensor | None = None, meta: dict | None = None) -> None:
        """If `affine` is given, use it. Else, if `affine` exists in the input tensor, use it. Else, use
        the default value. The same is true for `meta` and `transforms`."""
        self.set_initial_val("affine", affine, x, self.get_default_affine)
        self.set_initial_val("meta", meta, x, self.get_default_meta)

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
        meta_args = MetaObj.get_tensors_or_arrays(list(args) + list(kwargs.values()))
        ret._copy_meta(meta_args)
        return ret

    def get_default_affine(self) -> torch.Tensor:
        return torch.eye(4, device=self.device)

    def as_tensor(self) -> torch.Tensor:
        """
        Return the `MetaTensor` as a `torch.Tensor`.
        It is OS dependent as to whether this will be a deep copy or not.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.tensor(self)
