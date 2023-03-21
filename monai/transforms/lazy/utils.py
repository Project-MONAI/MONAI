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

import math

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
    if isinstance(pending_item, MetaMatrix):
        return pending_item.metadata

    if not isinstance(pending_item, dict):
        return {}
    ret = {
        LazyAttr.INTERP_MODE: pending_item.get(LazyAttr.INTERP_MODE, None),  # interpolation mode
        LazyAttr.PADDING_MODE: pending_item.get(LazyAttr.PADDING_MODE, None),  # padding mode
    }
    if LazyAttr.OUT_SHAPE in pending_item:
        ret[LazyAttr.OUT_SHAPE] = pending_item[LazyAttr.OUT_SHAPE]
    elif "shape_override" in pending_item:
        ret[LazyAttr.OUT_SHAPE] = pending_item["shape_override"]
    if LazyAttr.OUT_DTYPE in pending_item:
        ret[LazyAttr.OUT_DTYPE] = pending_item[LazyAttr.OUT_DTYPE]
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


def matrix_to_eulers(matrix):
    odd = 0 if (0+1) % 3 == 1 else 1
    i = 0
    j = (0 + 1 + odd) % 3
    k = (0 + 2 - odd) % 3
    print(odd, i, j, k)

    res_0 = math.atan2(matrix[j, k], matrix[k, k])
    c2 = np.linalg.norm(np.asarray([matrix[i, i], matrix[i, j]]))

    if odd == 0 and res_0 < 0 or odd == 1 and res_0 > 0:
        if res_0 > 0:
            res_0 -= torch.pi
        else:
            res_0 += torch.pi
        res_1 = math.atan2(-matrix[i, k], -c2)
    else:
        res_1 = math.atan2(-matrix[i, k], c2)

    s1 = np.sin(res_0)
    c1 = np.cos(res_0)
    res_2 = math.atan2(s1 * matrix[k, i] - c1 * matrix[j, i], c1 * matrix[j, j] - s1 * matrix[2, 1])

    if odd == 0:
        res_0 = -res_0
        res_1 = -res_1
        res_2 = -res_2

    return np.asarray([res_0, res_1, res_2])


def check_matrix(matrix, atol=1e-5):
    if not is_matrix_shaped(matrix):
        raise ValueError(f"'matrix' must be (3,3) or (4,4) but has shape {matrix.shape}")

    spatial_dims = matrix.shape[0] - 1
    unit_scale = True
    is_ortho = True
    for d in range(spatial_dims):
        vec = matrix[d, :-1]
        norm_vec = np.linalg.norm(vec)
        unit_scale = unit_scale and np.isclose(norm_vec, 1.0, atol)
        is_ortho = is_ortho and np.isclose(np.abs(vec).max(), norm_vec, atol)
    if spatial_dims == 2:
        is_unskewed = np.isclose(np.dot(matrix[0, :-1], matrix[1, :-1]), 0.0, atol)
    else:
        is_unskewed = np.isclose(np.dot(matrix[0, :-1], matrix[1, :-1]), 0.0, atol) and \
                      np.isclose(np.dot(matrix[0, :-1], matrix[2, :-1]), 0.0, atol) and \
                      np.isclose(np.dot(matrix[1, :-1], matrix[2, :-1]), 0.0, atol)

    return (
        is_ortho,
        unit_scale,
        is_unskewed,
    )


def check_unit_translate(matrix, src_image_shape, dst_image_shape):

    src_image_shape_ = src_image_shape[1:]
    dst_image_shape_ = dst_image_shape[1:]

    if len(src_image_shape_) != len(dst_image_shape_):
        raise ValueError("'src_image_shape' and 'dst_image_shape' must be sequences of the same length "
                         f"but are length {src_image_shape_} and {dst_image_shape_} respectively")

    for i, (s, d) in enumerate(zip(src_image_shape_, dst_image_shape_)):
        partial = matrix[i, -1] - np.floor(matrix[i, -1])
        if s % 2 == d % 2:
            if not (np.isclose(partial, 1.0, atol=1e-5) or np.isclose(partial, 0.0, atol=1e-5)):
                return False
        else:
            if not np.isclose(partial, 0.5, atol=1e-5):
                return False

    return True


# def check_axes(matrix):
#     is_ortho, is_scaled = check_matrix(matrix)
#
#     if not is_ortho:
#         raise ValueError(f"check_axes only accepts orthogonally aligned matrices (matrix is {matrix}")
#
#     x = matrix[0, :-1]
#     y = matrix[1, :-1]
#     z = matrix[2, :-1]
#     x_ind = np.argwhere(np.abs(x) > 1e-6)[0, 0]
#     y_ind = np.argwhere(np.abs(y) > 1e-6)[0, 0]
#     z_ind = np.argwhere(np.abs(z) > 1e-6)[0, 0]
#     return ((x_ind, 1 if x[x_ind] > 0.0 else -1),
#             (y_ind, 1 if y[y_ind] > 0.0 else -1),
#             (z_ind, 1 if z[z_ind] > 0.0 else -1))


def check_axes(matrix):
    if not is_matrix_shaped(matrix):
        raise ValueError(f"'matrix' must be (3,3) or (4,4) but has shape {matrix.shape}")

    spatial_dims = matrix.shape[0] - 1

    is_ortho, is_scaled, is_unskewed = check_matrix(matrix)

    if not is_ortho or not is_unskewed:
        raise ValueError(f"check_axes only accepts orthogonally aligned, unskewed matrices (matrix is {matrix}")

    indices = list()
    flips = list()
    for d in range(spatial_dims):
        vec = matrix[d, :-1]
        index = np.argwhere(np.abs(vec) > 1e-6)[0, 0]
        indices.append(index)
        if vec[index] < 0.0:
            flips.append(d)

    return tuple(flips), tuple(indices)


def get_scaling_factors(matrix):

    if not is_matrix_shaped(matrix):
        raise ValueError(f"'matrix' must be (3,3) or (4,4) but has shape {matrix.shape}")

    spatial_dims = matrix.shape[0] - 1

    is_ortho, is_scaled, is_unskewed = check_matrix(matrix)

    if not is_ortho or not is_unskewed:
        raise ValueError(f"check_axes only accepts orthogonally aligned, unskewed matrices (matrix is {matrix}")

    indices = list()
    for d in range(spatial_dims):
        vec = matrix[d, :-1]
        index = np.argwhere(np.abs(vec) > 1e-6)[0, 0]
        indices.append(index)

    zooms = list()
    for i_d, d in enumerate(indices):
        zooms.append(matrix[i_d, d])

    return tuple(zooms)


class Matrix:
    def __init__(self, matrix: NdarrayOrTensor):
        self.data = ensure_tensor(matrix)

    def __repr__(self):
        return f"Matrix<data={self.data if is_matrix_shaped(self.data) else self.data.shape}>"

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
            if len(matrix.shape) == 2:
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

    def invert(self):
        self.matrix.data = torch.inverse(self.matrix.data)

        def swap(d, k1, k2):
            d[k1], d[k2] = d[k2], d[k1]

        swap(self.metadata, LazyAttr.IN_DTYPE, LazyAttr.OUT_DTYPE)
        swap(self.metadata, LazyAttr.IN_SHAPE, LazyAttr.OUT_SHAPE)

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

    def invert(self):
        self.matrix.data = torch.inverse(self.matrix.data)
        self.metadata[LazyAttr.IN_DTYPE], self.metadata[LazyAttr.OUT_DTYPE] =\
            self.metadata[LazyAttr.OUT_DTYPE], self.metadata[LazyAttr.IN_DTYPE]
        self.metadata[LazyAttr.IN_SHAPE], self.metadata[LazyAttr.OUT_SHAPE] =\
            self.metadata[LazyAttr.OUT_SHAPE], self.metadata[LazyAttr.IN_SHAPE]

    def __repr__(self):
        return f"MetaMatrix<matrix={self.matrix}, metadata={self.metadata}>"
