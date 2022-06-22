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

import torch
from numpy import ndarray
from torch import Tensor

from monai.config.type_definitions import NdarrayOrTensor


def ifftn(ksp: NdarrayOrTensor, spatial_dims: int, is_complex: bool = True) -> Tensor:
    """
    Pytorch-based ifft for spatial_dims-dim signals.
    inputs:
        ksp: k-space data
        spatial_dims: number of spatial dimensions (e.g., is 2 for an image, and is 3 for a volume)
        is_complex: if True, then the last dimension of the input ksp is expected to be 2 (representing real and imaginary channels)
    """
    # handle numpy format
    isnp = False
    if isinstance(ksp, ndarray):
        ksp = torch.from_numpy(ksp)
        isnp = True

    # define spatial dims to perform ifftshift, fftshift, and ifft
    shift = tuple(range(-spatial_dims, 0))
    if is_complex:
        assert ksp.shape[-1] == 2
        shift = tuple(range(-spatial_dims - 1, -1))
    dims = tuple(range(-spatial_dims, 0))

    # apply ifft
    x = torch.fft.ifftshift(ksp, dim=shift)
    if is_complex:
        x = torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(x), dim=dims, norm="ortho"))
    else:
        x = torch.view_as_real(torch.fft.ifftn(x, dim=dims, norm="ortho"))
    out = torch.fft.fftshift(x, dim=shift)

    # handle numpy format
    if isnp:
        out = out.numpy()
    return out


def fftn(im: NdarrayOrTensor, spatial_dims: int, is_complex: bool = True) -> Tensor:
    """
    Pytorch-based fft for spatial_dims-dim signals.
    inputs:
        im: image
        spatial_dims: number of spatial dimensions (e.g., is 2 for an image, and is 3 for a volume)
        is_complex: if True, then the last dimension of the input im is expected to be 2 (representing real and imaginary channels)
    """
    # handle numpy format
    isnp = False
    if isinstance(im, ndarray):
        im = torch.from_numpy(im)
        isnp = True

    # define spatial dims to perform ifftshift, fftshift, and fft
    shift = tuple(range(-spatial_dims, 0))
    if is_complex:
        assert im.shape[-1] == 2
        shift = tuple(range(-spatial_dims - 1, -1))
    dims = tuple(range(-spatial_dims, 0))

    # apply fft
    x = torch.fft.ifftshift(im, dim=shift)
    if is_complex:
        x = torch.view_as_real(torch.fft.fftn(torch.view_as_complex(x), dim=dims, norm="ortho"))
    else:
        x = torch.view_as_real(torch.fft.fftn(x, dim=dims, norm="ortho"))
    out = torch.fft.fftshift(x, dim=shift)

    # handle numpy format
    if isnp:
        out = out.numpy()
    return out
