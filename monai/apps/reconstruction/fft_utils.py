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
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type


def ifftn(ksp: NdarrayOrTensor, spatial_dims: int, is_complex: bool = True) -> Tensor:
    """
    Pytorch-based ifft for spatial_dims-dim signals.
    Args:
        ksp: k-space data
        spatial_dims: number of spatial dimensions (e.g., is 2 for an image, and is 3 for a volume)
        is_complex: if True, then the last dimension of the input ksp is expected to be 2 (representing real and imaginary channels)
    Returns:
        out: output image (inverse fourier of ksp)
    """
    # handle numpy format
    isnp = False
    if isinstance(ksp, ndarray):
        ksp_t, *_ = convert_data_type(ksp, torch.Tensor)
        isnp = True
    else:
        ksp_t = ksp.clone()

    # define spatial dims to perform ifftshift, fftshift, and ifft
    shift = tuple(range(-spatial_dims, 0))
    if is_complex:
        assert ksp.shape[-1] == 2
        shift = tuple(range(-spatial_dims - 1, -1))
    dims = tuple(range(-spatial_dims, 0))

    # apply ifft
    x = torch.fft.ifftshift(ksp_t, dim=shift)
    if is_complex:
        x = torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(x), dim=dims, norm="ortho"))
    else:
        x = torch.view_as_real(torch.fft.ifftn(x, dim=dims, norm="ortho"))
    out_t = torch.fft.fftshift(x, dim=shift)

    # handle numpy format
    if isnp:
        out, *_ = convert_to_dst_type(src=out_t, dst=ksp)
    else:
        out = out_t.clone()
    return out


def fftn(im: NdarrayOrTensor, spatial_dims: int, is_complex: bool = True) -> Tensor:
    """
    Pytorch-based fft for spatial_dims-dim signals.
    Args:
        im: image
        spatial_dims: number of spatial dimensions (e.g., is 2 for an image, and is 3 for a volume)
        is_complex: if True, then the last dimension of the input im is expected to be 2 (representing real and imaginary channels)
    Returns:
        out: output kspace (fourier of im)
    """
    # handle numpy format
    isnp = False
    if isinstance(im, ndarray):
        im_t, *_ = convert_data_type(im, torch.Tensor)
        isnp = True
    else:
        im_t = im.clone()

    # define spatial dims to perform ifftshift, fftshift, and fft
    shift = tuple(range(-spatial_dims, 0))
    if is_complex:
        assert im.shape[-1] == 2
        shift = tuple(range(-spatial_dims - 1, -1))
    dims = tuple(range(-spatial_dims, 0))

    # apply fft
    x = torch.fft.ifftshift(im_t, dim=shift)
    if is_complex:
        x = torch.view_as_real(torch.fft.fftn(torch.view_as_complex(x), dim=dims, norm="ortho"))
    else:
        x = torch.view_as_real(torch.fft.fftn(x, dim=dims, norm="ortho"))
    out_t = torch.fft.fftshift(x, dim=shift)

    # handle numpy format
    if isnp:
        out, *_ = convert_to_dst_type(src=out_t, dst=im)
    else:
        out = out_t.clone()
    return out
