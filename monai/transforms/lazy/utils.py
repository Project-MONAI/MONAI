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

import numpy as np
import torch

import monai
from monai.config import NdarrayOrTensor
from monai.utils import LazyAttr, convert_to_tensor

__all__ = ["resample", "combine_transforms"]


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
        if isinstance(data, DisplacementField):
            return False
        if not hasattr(data, "shape") or len(data.shape) < 2:
            return False
        return data.shape[-1] in (3, 4) and data.shape[-2] in (3, 4) and data.shape[-1] == data.shape[-2]


class DisplacementField:
    """A class to represent a dense displacement field."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_ddf_shaped(data):
        """Check if the data is a DDF."""
        if isinstance(data, DisplacementField):
            return True
        if isinstance(data, Affine):
            return False
        if not hasattr(data, "shape") or len(data.shape) < 3:
            return False
        return not Affine.is_affine_shaped(data)


def combine_transforms(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Given transforms A and B to be applied to x, return the combined transform (AB), so that A(B(x)) becomes AB(x)"""
    if Affine.is_affine_shaped(left) and Affine.is_affine_shaped(right):  # linear transforms
        left = convert_to_tensor(left.data if isinstance(left, Affine) else left, wrap_sequence=True)
        right = convert_to_tensor(right.data if isinstance(right, Affine) else right, wrap_sequence=True)
        return torch.matmul(left, right)
    if DisplacementField.is_ddf_shaped(left) and DisplacementField.is_ddf_shaped(
        right
    ):  # adds DDFs, do we need metadata if metatensor input?
        left = convert_to_tensor(left.data if isinstance(left, DisplacementField) else left, wrap_sequence=True)
        right = convert_to_tensor(right.data if isinstance(right, DisplacementField) else right, wrap_sequence=True)
        return left + right
    raise NotImplementedError


def affine_from_pending(pending_item):
    """Extract the affine matrix from a pending transform item."""
    if isinstance(pending_item, (torch.Tensor, np.ndarray)):
        return pending_item
    if isinstance(pending_item, dict):
        return pending_item[LazyAttr.AFFINE]
    return pending_item


def kwargs_from_pending(pending_item):
    """Extract kwargs from a pending transform item."""
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


def is_compatible_apply_kwargs(kwargs_1, kwargs_2):
    """Check if two sets of kwargs are compatible (to be combined in `apply`)."""
    return True


def resample(data: torch.Tensor, matrix: NdarrayOrTensor, spatial_size, kwargs: dict | None = None):
    """
    This is a minimal implementation of resample that always uses SpatialResample.
    `kwargs` supports "lazy_dtype", "lazy_padding_mode", "lazy_interpolation_mode", "lazy_dtype", "lazy_align_corners".

    See Also:
        :py:class:`monai.transforms.SpatialResample`
    """
    if not Affine.is_affine_shaped(matrix):
        raise NotImplementedError("calling dense grid resample API not implemented")
    kwargs = {} if kwargs is None else kwargs
    init_kwargs = {
        "dtype": kwargs.pop(LazyAttr.DTYPE, data.dtype),
        "align_corners": kwargs.pop(LazyAttr.ALIGN_CORNERS, None),
    }
    img = convert_to_tensor(data=data, track_meta=monai.data.get_track_meta())
    init_affine = monai.data.to_affine_nd(len(matrix) - 1, img.affine)
    call_kwargs = {
        "spatial_size": img.peek_pending_shape() if spatial_size is None else spatial_size,
        "dst_affine": init_affine @ monai.utils.convert_to_dst_type(matrix, init_affine)[0],
        "mode": kwargs.pop(LazyAttr.INTERP_MODE, None),
        "padding_mode": kwargs.pop(LazyAttr.PADDING_MODE, None),
    }
    resampler = monai.transforms.SpatialResample(**init_kwargs)
    # resampler.lazy_evaluation = False
    with resampler.trace_transform(False):  # don't track this transform in `data`
        return resampler(img=img, **call_kwargs)
