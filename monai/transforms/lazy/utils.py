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

from typing import Union

import numpy as np
import torch

# import monai
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import LazyAttr, convert_to_tensor

__all__ = ["combine_transforms"]


class AffineMatrix:
    """A class to represent an affine transform matrix."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_affine_shaped(data):
        """Check if the data is an affine matrix."""
        if isinstance(data, AffineMatrix):
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
        if isinstance(data, AffineMatrix):
            return False
        if not hasattr(data, "shape") or len(data.shape) < 3:
            return False
        return not AffineMatrix.is_affine_shaped(data)


def combine_transforms(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Given transforms A and B to be applied to x, return the combined transform (AB), so that A(B(x)) becomes AB(x)"""
    if AffineMatrix.is_affine_shaped(left) and AffineMatrix.is_affine_shaped(right):  # linear transforms
        left = convert_to_tensor(left.data if isinstance(left, AffineMatrix) else left, wrap_sequence=True)
        right = convert_to_tensor(right.data if isinstance(right, AffineMatrix) else right, wrap_sequence=True)
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
    if isinstance(pending_item, MetaMatrix):
        return pending_item.matrix.data
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
    elif "shape_override" in pending_item:
        ret[LazyAttr.SHAPE] = pending_item["shape_override"]
    if LazyAttr.DTYPE in pending_item:
        ret[LazyAttr.DTYPE] = pending_item[LazyAttr.DTYPE]
    return ret


def is_compatible_apply_kwargs(kwargs_1, kwargs_2):
    """Check if two sets of kwargs are compatible (to be combined in `apply`)."""
    return True


def ensure_tensor(data: NdarrayOrTensor):
    if isinstance(data, torch.Tensor):
        return data

    return torch.as_tensor(data)


def is_matrix_shaped(data):

    return (
        len(data.shape) == 2 and data.shape[0] in (3, 4) and data.shape[1] in (3, 4) and data.shape[0] == data.shape[1]
    )


# this will conflict with PR Replacement Apply and Resample #5436
def is_grid_shaped(data):

    return len(data.shape) == 3 and data.shape[0] == 3 or len(data.shape) == 4 and data.shape[0] == 4


class Matrix:
    def __init__(self, matrix: NdarrayOrTensor):
        self.data = ensure_tensor(matrix)

    # def __matmul__(self, other):
    #     if isinstance(other, Matrix):
    #         other_matrix = other.data
    #     else:
    #         other_matrix = other
    #     return self.data @ other_matrix
    #
    # def __rmatmul__(self, other):
    #     return other.__matmul__(self.data)


# this will conflict with PR Replacement Apply and Resample #5436
class Grid:
    def __init__(self, grid):
        self.data = ensure_tensor(grid)

    # def __matmul__(self, other):
    #     raise NotImplementedError()


# this will conflict with PR Replacement Apply and Resample #5436
class MetaMatrix:
    def __init__(
            self, matrix: Union[NdarrayOrTensor, Matrix, Grid],
            metadata: dict | None = None
    ):
        if not isinstance(matrix, (Matrix, Grid)):
            if matrix.shape == 2:
                if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] not in (3, 4):
                    raise ValueError(
                        "If 'matrix' is passed a numpy ndarray/torch Tensor, it must"
                        f" be 3x3 or 4x4 ('matrix' has has shape {matrix.shape})"
                    )
            matrix_ = Matrix(matrix)
        else:
            matrix_ = matrix
        self.matrix = matrix_

        self.metadata = metadata or {}

    def __matmul__(self, other):
        if isinstance(other, MetaMatrix):
            other_ = other.matrix
        else:
            other_ = other
        return MetaMatrix(self.matrix @ other_)

    def __rmatmul__(self, other):
        if isinstance(other, MetaMatrix):
            other_ = other.matrix
        else:
            other_ = other
        return MetaMatrix(other_ @ self.matrix)
