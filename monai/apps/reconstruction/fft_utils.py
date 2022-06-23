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

from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type


def ifftn_centered(ksp: NdarrayOrTensor, spatial_dims: int, is_complex: bool = True) -> NdarrayOrTensor:
    """
    Pytorch-based ifft for spatial_dims-dim signals.
    This is equivalent to do fft in numpy based on numpy.fft.ifftn, numpy.fft.fftshift, and numpy.fft.ifft.shift

    Args:
        ksp: k-space data
        spatial_dims: number of spatial dimensions (e.g., is 2 for an image, and is 3 for a volume)
        is_complex: if True, then the last dimension of the input ksp is expected to be 2 (representing real and imaginary channels)

    Returns:
        Union[ndarray,Tensor] "out" which is the output image (inverse fourier of ksp)

    Example:

        .. code-block:: python

            import torch
            ksp = torch.ones(1,3,3,2) # the last dim belongs to real/imaginary parts
            # output1 and output2 will be identical
            output1 = torch.fft.ifftn(torch.view_as_complex(torch.fft.ifftshift(ksp,dim=(-3,-2))), dim=(-2,-1), norm="ortho")
            output1 = torch.fft.fftshift( torch.view_as_real(output1), dim=(-3,-2) )

            output2 = ifftn_centered(ksp, spatial_dims=2, is_complex=True)
    """
    # handle numpy format
    ksp_t, *_ = convert_data_type(ksp, torch.Tensor)

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
    out, *_ = convert_to_dst_type(src=out_t, dst=ksp)
    return out


def fftn_centered(im: NdarrayOrTensor, spatial_dims: int, is_complex: bool = True) -> NdarrayOrTensor:
    """
    Pytorch-based fft for spatial_dims-dim signals.
    This is equivalent to do ifft in numpy based on numpy.fft.fftn, numpy.fft.fftshift, and numpy.fft.ifft.shift

    Args:
        im: image
        spatial_dims: number of spatial dimensions (e.g., is 2 for an image, and is 3 for a volume)
        is_complex: if True, then the last dimension of the input im is expected to be 2 (representing real and imaginary channels)

    Returns:
        Union[ndarray,Tensor] "out" which is the output kspace (fourier of im)

    Example:

        .. code-block:: python

            import torch
            im = torch.ones(1,3,3,2) # the last dim belongs to real/imaginary parts
            # output1 and output2 will be identical
            output1 = torch.fft.fftn(torch.view_as_complex(torch.fft.ifftshift(im,dim=(-3,-2))), dim=(-2,-1), norm="ortho")
            output1 = torch.fft.fftshift( torch.view_as_real(output1), dim=(-3,-2) )

            output2 = fftn_centered(im, spatial_dims=2, is_complex=True)
    """
    # handle numpy format
    im_t, *_ = convert_data_type(im, torch.Tensor)

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
    out, *_ = convert_to_dst_type(src=out_t, dst=im)
    return out
