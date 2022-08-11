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

from typing import List

import torch
from torch import Tensor



def roll_1d(x: Tensor, shift: int, shift_dim: int) -> Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: input data (k-space or image) that can be
            1) real-valued: the shape is (C,H,W) for 2D spatial inputs and (C,H,W,D) for 3D, or
            2) complex-valued: the shape is (C,H,W,2) for 2D spatial data and (C,H,W,D,2) for 3D. C is the number of channels.
        shift: the amount of shift along each of shift_dims dimension
        shift_dim: the dimension over which the shift is applied

    Returns:
        1d-shifted version of x

    Note:
        This function is called when fftshift and ifftshift are not available in the running pytorch version
    """
    shift = shift % x.size(shift_dim)
    if shift == 0:
        return x

    left = x.narrow(shift_dim, 0, x.size(shift_dim) - shift)
    right = x.narrow(shift_dim, x.size(shift_dim) - shift, shift)

    return torch.cat((right, left), dim=shift_dim)


def roll(x: Tensor, shift: List[int], shift_dims: List[int]) -> Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors

    Args:
        x: input data (k-space or image) that can be
            1) real-valued: the shape is (C,H,W) for 2D spatial inputs and (C,H,W,D) for 3D, or
            2) complex-valued: the shape is (C,H,W,2) for 2D spatial data and (C,H,W,D,2) for 3D. C is the number of channels.
        shift: the amount of shift along each of shift_dims dimensions
        shift_dims: dimensions over which the shift is applied

    Returns:
        shifted version of x

    Note:
        This function is called when fftshift and ifftshift are not available in the running pytorch version
    """
    if len(shift) != len(shift_dims):
        raise ValueError(f"len(shift) != len(shift_dims), got f{len(shift)} and f{len(shift_dims)}.")
    for s, d in zip(shift, shift_dims):
        x = roll_1d(x, s, d)
    return x


def fftshift(x: Tensor, shift_dims: List[int]) -> Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: input data (k-space or image) that can be
            1) real-valued: the shape is (C,H,W) for 2D spatial inputs and (C,H,W,D) for 3D, or
            2) complex-valued: the shape is (C,H,W,2) for 2D spatial data and (C,H,W,D,2) for 3D. C is the number of channels.
        shift_dims: dimensions over which the shift is applied

    Returns:
        fft-shifted version of x

    Note:
        This function is called when fftshift is not available in the running pytorch version
    """
    shift = [0] * len(shift_dims)
    for i, dim_num in enumerate(shift_dims):
        shift[i] = x.shape[dim_num] // 2
    return roll(x, shift, shift_dims)


def ifftshift(x: Tensor, shift_dims: List[int]) -> Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: input data (k-space or image) that can be
            1) real-valued: the shape is (C,H,W) for 2D spatial inputs and (C,H,W,D) for 3D, or
            2) complex-valued: the shape is (C,H,W,2) for 2D spatial data and (C,H,W,D,2) for 3D. C is the number of channels.
        shift_dims: dimensions over which the shift is applied

    Returns:
        ifft-shifted version of x

    Note:
        This function is called when ifftshift is not available in the running pytorch version
    """
    shift = [0] * len(shift_dims)
    for i, dim_num in enumerate(shift_dims):
        shift[i] = (x.shape[dim_num] + 1) // 2
    return roll(x, shift, shift_dims)


def ifftn_centered_t(ksp: Tensor, spatial_dims: int, is_complex: bool = True) -> Tensor:
    """
    Pytorch-based ifft for spatial_dims-dim signals. "centered" means this function automatically takes care
    of the required ifft and fft shifts.
    This is equivalent to do fft in numpy based on numpy.fft.ifftn, numpy.fft.fftshift, and numpy.fft.ifftshift

    Args:
        ksp: k-space data that can be
            1) real-valued: the shape is (C,H,W) for 2D spatial inputs and (C,H,W,D) for 3D, or
            2) complex-valued: the shape is (C,H,W,2) for 2D spatial data and (C,H,W,D,2) for 3D. C is the number of channels.
        spatial_dims: number of spatial dimensions (e.g., is 2 for an image, and is 3 for a volume)
        is_complex: if True, then the last dimension of the input ksp is expected to be 2 (representing real and imaginary channels)

    Returns:
        "out" which is the output image (inverse fourier of ksp)

    Example:

        .. code-block:: python

            import torch
            ksp = torch.ones(1,3,3,2) # the last dim belongs to real/imaginary parts
            # output1 and output2 will be identical
            output1 = torch.fft.ifftn(torch.view_as_complex(torch.fft.ifftshift(ksp,dim=(-3,-2))), dim=(-2,-1), norm="ortho")
            output1 = torch.fft.fftshift( torch.view_as_real(output1), dim=(-3,-2) )

            output2 = ifftn_centered(ksp, spatial_dims=2, is_complex=True)
    """
    # define spatial dims to perform ifftshift, fftshift, and ifft
    shift = list(range(-spatial_dims, 0))
    if is_complex:
        if ksp.shape[-1] != 2:
            raise ValueError(f"ksp.shape[-1] is not 2 ({ksp.shape[-1]}).")
        shift = list(range(-spatial_dims - 1, -1))
    dims = list(range(-spatial_dims, 0))

    x = ifftshift(ksp, shift)

    if is_complex:
        x = torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(x), dim=dims, norm="ortho"))
    else:
        x = torch.view_as_real(torch.fft.ifftn(x, dim=dims, norm="ortho"))

    out: Tensor = fftshift(x, shift)

    return out


def fftn_centered_t(im: Tensor, spatial_dims: int, is_complex: bool = True) -> Tensor:
    """
    Pytorch-based fft for spatial_dims-dim signals. "centered" means this function automatically takes care
    of the required ifft and fft shifts.
    This is equivalent to do ifft in numpy based on numpy.fft.fftn, numpy.fft.fftshift, and numpy.fft.ifftshift

    Args:
        im: image that can be
            1) real-valued: the shape is (C,H,W) for 2D spatial inputs and (C,H,W,D) for 3D, or
            2) complex-valued: the shape is (C,H,W,2) for 2D spatial data and (C,H,W,D,2) for 3D. C is the number of channels.
        spatial_dims: number of spatial dimensions (e.g., is 2 for an image, and is 3 for a volume)
        is_complex: if True, then the last dimension of the input im is expected to be 2 (representing real and imaginary channels)

    Returns:
        "out" which is the output kspace (fourier of im)

    Example:

        .. code-block:: python

            import torch
            im = torch.ones(1,3,3,2) # the last dim belongs to real/imaginary parts
            # output1 and output2 will be identical
            output1 = torch.fft.fftn(torch.view_as_complex(torch.fft.ifftshift(im,dim=(-3,-2))), dim=(-2,-1), norm="ortho")
            output1 = torch.fft.fftshift( torch.view_as_real(output1), dim=(-3,-2) )

            output2 = fftn_centered(im, spatial_dims=2, is_complex=True)
    """
    # define spatial dims to perform ifftshift, fftshift, and fft
    shift = list(range(-spatial_dims, 0))
    if is_complex:
        if im.shape[-1] != 2:
            raise ValueError(f"img.shape[-1] is not 2 ({im.shape[-1]}).")
        shift = list(range(-spatial_dims - 1, -1))
    dims = list(range(-spatial_dims, 0))

    x = ifftshift(im, shift)

    if is_complex:
        x = torch.view_as_real(torch.fft.fftn(torch.view_as_complex(x), dim=dims, norm="ortho"))
    else:
        x = torch.view_as_real(torch.fft.fftn(x, dim=dims, norm="ortho"))

    out: Tensor = fftshift(x, shift)

    return out
