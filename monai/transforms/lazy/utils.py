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

from typing import Optional

import numpy as np
import torch

import monai
from monai.config import NdarrayOrTensor
from monai.utils import LazyAttr

__all__ = ["resample", "matmul"]


class Affine:
    """A class to represent an affine transform matrix."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_affine_shaped(data):
        """Check if the data is an affine matrix."""
        if isinstance(data, Affine):
            return True
        if isinstance(data, DDF):
            return False
        if not hasattr(data, "shape") or len(data.shape) < 2:
            return False
        return data.shape[-1] in (3, 4) and data.shape[-2] in (3, 4) and data.shape[-1] == data.shape[-2]


class DDF:
    """A class to represent a dense displacement field."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_ddf_shaped(data):
        """Check if the data is a DDF."""
        if isinstance(data, DDF):
            return True
        if isinstance(data, Affine):
            return False
        if not hasattr(data, "shape") or len(data.shape) < 3:
            return False
        return not Affine.is_affine_shaped(data)


def matmul(left: torch.Tensor, right: torch.Tensor):
    if Affine.is_affine_shaped(left) and Affine.is_affine_shaped(right):  # linear transforms
        if isinstance(left, Affine):
            left = left.data
        if isinstance(right, Affine):
            right = right.data
        return torch.matmul(left, right)
    if DDF.is_ddf_shaped(left) and DDF.is_ddf_shaped(right):  # adds DDFs
        return left + right
    raise NotImplementedError


def mat_from_pending(pending_item):
    if isinstance(pending_item, (torch.Tensor, np.ndarray)):
        return pending_item
    if isinstance(pending_item, dict):
        return pending_item[LazyAttr.AFFINE]
    return pending_item


def kwargs_from_pending(pending_item):
    if not isinstance(pending_item, dict):
        return {}
    ret = {
        LazyAttr.INTERP_MODE: pending_item.get(LazyAttr.INTERP_MODE, None),  # interpolation mode
        LazyAttr.PADDING_MODE: pending_item.get(LazyAttr.PADDING_MODE, None),  # padding mode
    }
    if LazyAttr.SHAPE in pending_item:
        ret[LazyAttr.SHAPE] = pending_item[LazyAttr.SHAPE]
    if LazyAttr.DTYPE in pending_item:
        ret[LazyAttr.DTYPE] = pending_item[LazyAttr.DTYPE]
    return ret


def is_compatible_kwargs(kwargs_1, kwargs_2):
    return True


def resample(data: torch.Tensor, matrix: NdarrayOrTensor, kwargs: Optional[dict] = None):
    """
    This is a minimal implementation of resample that always uses Affine.
    """
    if not Affine.is_affine_shaped(matrix):
        raise NotImplementedError("calling dense grid resample API not implemented")
    kwargs = {} if kwargs is None else kwargs
    init_kwargs = {
        "spatial_size": kwargs.pop(LazyAttr.SHAPE, data.shape)[1:],
        "dtype": kwargs.pop(LazyAttr.DTYPE, data.dtype),
    }
    call_kwargs = {
        "mode": kwargs.pop(LazyAttr.INTERP_MODE, None),
        "padding_mode": kwargs.pop(LazyAttr.PADDING_MODE, None),
    }
    resampler = monai.transforms.Affine(affine=matrix, image_only=True, **init_kwargs)
    with resampler.trace_transform(False):  # don't track this transform in `data`
        return resampler(img=data, **call_kwargs)
