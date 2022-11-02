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

# placeholder to be replaced by MetaMatrix in Apply And Resample PR #5436
from monai.transforms.utils import _create_rotate, _create_shear, _create_scale, _create_translate

from monai.utils import TransformBackends


# this will conflict with PR Replacement Apply and Resample #5436
class MetaMatrix:

    def __init__(self):
        raise NotImplementedError()


# this will conflict with PR Replacement Apply and Resample #5436
class MatrixFactory:

    def __init__(self,
                 dims: int,
                 backend: TransformBackends,
                 device: Optional[torch.device] = None):

        if backend == TransformBackends.NUMPY:
            if device is not None:
                raise ValueError("'device' must be None with TransformBackends.NUMPY")
            self._device = None
            self._sin = lambda th: np.sin(th, dtype=np.float32)
            self._cos = lambda th: np.cos(th, dtype=np.float32)
            self._eye = lambda th: np.eye(th, dtype=np.float32)
            self._diag = lambda th: np.diag(th).astype(np.float32)
        else:
            if device is None:
                raise ValueError("'device' must be set with TransformBackends.TORCH")
            self._device = device
            self._sin = lambda th: torch.sin(torch.as_tensor(th,
                                                             dtype=torch.float32,
                                                             device=self._device))
            self._cos = lambda th: torch.cos(torch.as_tensor(th,
                                                             dtype=torch.float32,
                                                             device=self._device))
            self._eye = lambda rank: torch.eye(rank,
                                               device=self._device,
                                               dtype=torch.float32);
            self._diag = lambda size: torch.diag(torch.as_tensor(size,
                                                                 device=self._device,
                                                                 dtype=torch.float32))

        self._backend = backend
        self._dims = dims

    @staticmethod
    def from_tensor(data):
        return MatrixFactory(len(data.shape)-1,
                             get_backend_from_tensor_like(data),
                             get_device_from_tensor_like(data))

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


# this will conflict with PR Replacement Apply and Resample #5436
def apply_align_corners(matrix, spatial_size, factory):
    inflated_spatial_size = tuple(s + 1 for s in spatial_size)
    scale_factors = tuple(s / i for s, i in zip(spatial_size, inflated_spatial_size))
    scale_mat = factory.scale(scale_factors)
    return matmul(scale_mat, matrix)