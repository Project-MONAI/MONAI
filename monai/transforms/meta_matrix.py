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

from typing import Optional, Union

import numpy as np
import torch

from monai.config import NdarrayOrTensor


def is_matrix_shaped(data):

    return (
        len(data.shape) == 2 and data.shape[0] in (3, 4) and data.shape[1] in (3, 4) and data.shape[0] == data.shape[1]
    )


def is_grid_shaped(data):

    return len(data.shape) == 3 and data.shape[0] == 3 or len(data.shape) == 4 and data.shape[0] == 4


class MatrixFactory:
    pass


def ensure_tensor(data: NdarrayOrTensor):
    if isinstance(data, torch.Tensor):
        return data

    return torch.as_tensor(data)


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


class Grid:
    def __init__(self, grid):
        self.data = ensure_tensor(grid)

    # def __matmul__(self, other):
    #     raise NotImplementedError()


class MetaMatrix:
    def __init__(self, matrix: Union[NdarrayOrTensor, Matrix, Grid], metadata: Optional[dict] = None):

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


def matmul(
    left: Union[MetaMatrix, Grid, Matrix, NdarrayOrTensor], right: Union[MetaMatrix, Grid, Matrix, NdarrayOrTensor]
):
    matrix_types = (MetaMatrix, Grid, Matrix, torch.Tensor, np.ndarray)

    if not isinstance(left, matrix_types):
        raise TypeError(f"'left' must be one of {matrix_types} but is {type(left)}")
    if not isinstance(right, matrix_types):
        raise TypeError(f"'second' must be one of {matrix_types} but is {type(right)}")

    left_ = left.matrix if isinstance(left, MetaMatrix) else left
    right_ = right.matrix if isinstance(right, MetaMatrix) else right

    # TODO: it might be better to not return a metamatrix, unless we pass in the resulting
    # metadata also
    put_in_metamatrix = isinstance(left, MetaMatrix) or isinstance(right, MetaMatrix)

    put_in_grid = isinstance(left, Grid) or isinstance(right, Grid)

    put_in_matrix = isinstance(left, Matrix) or isinstance(right, Matrix)
    put_in_matrix = False if put_in_grid is True else put_in_matrix

    promote_to_tensor = not (isinstance(left_, np.ndarray) and isinstance(right_, np.ndarray))

    left_raw = left_.data if isinstance(left_, (Matrix, Grid)) else left_
    right_raw = right_.data if isinstance(right_, (Matrix, Grid)) else right_

    if promote_to_tensor:
        left_raw = torch.as_tensor(left_raw)
        right_raw = torch.as_tensor(right_raw)

    if isinstance(left_, Grid):
        if isinstance(right_, Grid):
            raise RuntimeError("Unable to matrix multiply two Grids")
        else:
            result = matmul_grid_matrix(left_raw, right_raw)
    else:
        if isinstance(right_, Grid):
            result = matmul_matrix_grid(left_raw, right_raw)
        else:
            result = matmul_matrix_matrix(left_raw, right_raw)

    if put_in_grid:
        result = Grid(result)
    elif put_in_matrix:
        result = Matrix(result)

    if put_in_metamatrix:
        result = MetaMatrix(result)

    return result


def matmul_matrix_grid(left: NdarrayOrTensor, right: NdarrayOrTensor):
    if not is_matrix_shaped(left):
        raise ValueError(f"'left' should be a 2D or 3D homogenous matrix but has shape {left.shape}")

    if not is_grid_shaped(right):
        raise ValueError(
            "'right' should be a 3D array with shape[0] == 2 or a "
            f"4D array with shape[0] == 3 but has shape {right.shape}"
        )

    # flatten the grid to take advantage of torch batch matrix multiply
    right_flat = right.reshape(right.shape[0], -1)
    result_flat = left @ right_flat
    # restore the grid shape
    result = result_flat.reshape((-1,) + result_flat.shape[1:])
    return result


def matmul_grid_matrix(left: NdarrayOrTensor, right: NdarrayOrTensor):
    if not is_grid_shaped(left):
        raise ValueError(
            "'left' should be a 3D array with shape[0] == 2 or a "
            f"4D array with shape[0] == 3 but has shape {left.shape}"
        )

    if not is_matrix_shaped(right):
        raise ValueError(f"'right' should be a 2D or 3D homogenous matrix but has shape {right.shape}")

    try:
        inv_matrix = torch.inverse(right)
    except RuntimeError:
        # the matrix is not invertible, so we will have to perform a slow grid to matrix operation
        return matmul_grid_matrix_slow(left, right)

    # invert the matrix and swap the arguments, taking advantage of
    # matrix @ vector == vector_transposed @ matrix_inverse
    return matmul_matrix_grid(torch.inverse(right), left)


def matmul_grid_matrix_slow(left: NdarrayOrTensor, right: NdarrayOrTensor):
    if not is_grid_shaped(left):
        raise ValueError(
            "'left' should be a 3D array with shape[0] == 2 or a "
            f"4D array with shape[0] == 3 but has shape {left.shape}"
        )

    if not is_matrix_shaped(right):
        raise ValueError(f"'right' should be a 2D or 3D homogenous matrix but has shape {right.shape}")

    flat_left = left.reshape(left.shape[0], -1)
    result_flat = torch.zeros_like(flat_left)
    for i in range(flat_left.shape[1]):
        vector = flat_left[:, i][None, :]
        result_vector = vector @ right
        result_flat[:, i] = result_vector[0, :]

    # restore the grid shape
    result = result_flat.reshape((-1,) + result_flat.shape[1:])
    return result


def matmul_matrix_matrix(left: NdarrayOrTensor, right: NdarrayOrTensor):
    return left @ right
