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

from monai.utils.enums import TransformBackends
from monai.transforms.utils import (_create_rotate, _create_scale, _create_shear,
                                    _create_translate)

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
                                                             dtype=torch.float32,
                                                             device=self._device))
            self._cos = lambda th: torch.cos(torch.as_tensor(th,
                                                             dtype=torch.float32,
                                                             device=self._device))
            self._eye = lambda rank: torch.eye(rank, device=self._device);
            self._diag = lambda size: torch.diag(torch.as_tensor(size, device=self._device))

        self._backend = backend
        self._dims = dims

    def identity(self):
        return self._eye(self._dims + 1)

    def rotate_euler(self, radians: Union[Sequence[float], float]):
        return _create_rotate(self._dims, radians, self._sin, self._cos, self._eye)

    def shear(self, coefs: Union[Sequence[float], float]):
        return _create_shear(self._dims, coefs, self._eye)

    def scale(self, factors: Union[Sequence[float], float]):
        return _create_scale(self._dims, factors, self._diag)

    def translate(self, offsets: Union[Sequence[float], float]):
        return _create_translate(self._dims, offsets, self._eye)


class Mapping:

    def __init__(self, matrix):
        self._matrix = matrix

    def apply(self, other):
        return Mapping(other @ self._matrix)


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
