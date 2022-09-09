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

from typing import Optional, Sequence, Union

import numpy as np

import torch
from monai.config import NdarrayOrTensor

from monai.utils.enums import TransformBackends
from monai.transforms.utils import (_create_rotate, _create_scale, _create_shear,
                                    _create_translate, _create_rotate_90, _create_flip)
from monai.utils.misc import get_backend_from_data, get_device_from_data


def ensure_tensor(data: NdarrayOrTensor):
    if isinstance(data, torch.Tensor):
        return data

    return torch.as_tensor(data)


class MatrixFactory:

    def __init__(self,
                 dims: int,
                 backend: TransformBackends,
                 device: Optional[torch.device] = None):

        if backend == TransformBackends.NUMPY:
            if device is not None:
                raise ValueError("'device' must be None with TransformBackends.NUMPY")
            self._device = None
            self._sin = np.sin
            self._cos = np.cos
            self._eye = np.eye
            self._diag = np.diag
        else:
            if device is None:
                raise ValueError("'device' must be set with TransformBackends.TORCH")
            self._device = device
            self._sin = lambda th: torch.sin(torch.as_tensor(th,
                                                             dtype=torch.float64,
                                                             device=self._device))
            self._cos = lambda th: torch.cos(torch.as_tensor(th,
                                                             dtype=torch.float64,
                                                             device=self._device))
            self._eye = lambda rank: torch.eye(rank,
                                               device=self._device,
                                               dtype=torch.float64);
            self._diag = lambda size: torch.diag(torch.as_tensor(size,
                                                                 device=self._device,
                                                                 dtype=torch.float64))

        self._backend = backend
        self._dims = dims

    @staticmethod
    def from_tensor(data):
        return MatrixFactory(len(data.shape)-1,
                             get_backend_from_data(data),
                             get_device_from_data(data))

    def identity(self):
        matrix = self._eye(self._dims + 1)
        return MetaMatrix(matrix, {})

    def rotate_euler(self, radians: Union[Sequence[float], float], **extra_args):
        matrix = _create_rotate(self._dims, radians, self._sin, self._cos, self._eye)
        return MetaMatrix(matrix, extra_args)

    def rotate_90(self, rotations, axis, **extra_args):
        matrix = _create_rotate_90(self._dims, rotations, axis)
        return MetaMatrix(matrix, extra_args)

    def flip(self, axis, **extra_args):
        matrix = _create_flip(self._dims, axis, self._eye)
        return MetaMatrix(matrix, extra_args)

    def shear(self, coefs: Union[Sequence[float], float], **extra_args):
        matrix = _create_shear(self._dims, coefs, self._eye)
        return MetaMatrix(matrix, extra_args)

    def scale(self, factors: Union[Sequence[float], float], **extra_args):
        matrix = _create_scale(self._dims, factors, self._diag)
        return MetaMatrix(matrix, extra_args)

    def translate(self, offsets: Union[Sequence[float], float], **extra_args):
        matrix = _create_translate(self._dims, offsets, self._eye)
        return MetaMatrix(matrix, extra_args)


class Mapping:

    def __init__(self, matrix):
        self._matrix = matrix

    def apply(self, other):
        return Mapping(other @ self._matrix)


class Dimensions:

    def __init__(self, flips, permutes):
        raise NotImplementedError()

    def __matmul__(self, other):
        raise NotImplementedError()

    def __rmatmul__(self, other):
        raise NotImplementedError()


class Matrix:

    def __init__(self, matrix: NdarrayOrTensor):
        self.matrix = ensure_tensor(matrix)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            other_matrix = other.matrix
        else:
            other_matrix = other
        return self.matrix @ other_matrix

    def __rmatmul__(self, other):
        return other.__matmul__(self.matrix)


# TODO: remove if the existing Grid is fine for our purposes
class Grid:
    def __init__(self, grid):
        raise NotImplementedError()

    def __matmul__(self, other):
        raise NotImplementedError()


class MetaMatrix:

    def __init__(self, matrix, metadata=None):
        if not isinstance(matrix, (Matrix, Grid)):
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


class MappingStack:
    """
    This class keeps track of a series of mappings and apply them / calculate their inverse (if
    mappings are invertible). Mapping stacks are used to generate a mapping that gets applied during a `Resample` /
    `Resampled` transform.

    A mapping is one of:
    - a description of a change to a numpy array that only requires index manipulation instead of an actual resample.
    - a homogeneous matrix representing a geometric transform to be applied during a resample
    - a field representing a deformation to be applied during a resample
    """

    def __init__(self, factory: MatrixFactory):
        self.factory = factory
        self.stack = []
        self.applied_stack = []

    def push(self, mapping):
        self.stack.append(mapping)

    def pop(self):
        raise NotImplementedError()

    def transform(self):
        m = Mapping(self.factory.identity())
        for t in self.stack:
            m = m.apply(t)
        return m
