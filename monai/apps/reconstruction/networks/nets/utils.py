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
"""
This script contains utility functions for developing new networks/blocks in PyTorch.
"""

import math
from typing import Tuple

from torch import Tensor
from torch.nn import functional as F

from monai.apps.reconstruction.complex_utils import complex_conj_t, complex_mul_t
from monai.networks.blocks.fft_utils_t import fftn_centered_t, ifftn_centered_t


def reshape_complex_to_channel_dim(x: Tensor) -> Tensor:  # type: ignore
    """
    Swaps the complex dimension with the channel dimension so that the network treats real/imaginary
    parts as two separate channels.

    Args:
        x: input of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data

    Returns:
        output of shape (B,C*2,H,W) for 2D data or (B,C*2,H,W,D) for 3D data
    """
    if x.shape[-1] != 2:
        raise ValueError(f"last dim must be 2, but x.shape[-1] is {x.shape[-1]}.")

    if len(x.shape) == 5:  # this is 2D
        b, c, h, w, two = x.shape
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

    elif len(x.shape) == 6:  # this is 3D
        b, c, h, w, d, two = x.shape
        return x.permute(0, 5, 1, 2, 3, 4).contiguous().view(b, 2 * c, h, w, d)

    else:
        raise ValueError(f"only 2D (B,C,H,W,2) and 3D (B,C,H,W,D,2) data are supported but x has shape {x.shape}")


def reshape_channel_complex_to_last_dim(x: Tensor) -> Tensor:  # type: ignore
    """
    Swaps the complex dimension with the channel dimension so that the network output has 2 as its last dimension

    Args:
        x: input of shape (B,C*2,H,W) for 2D data or (B,C*2,H,W,D) for 3D data

    Returns:
        output of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data
    """
    if x.shape[1] % 2 != 0:
        raise ValueError(f"channel dimension should be even but ({x.shape[1]}) is odd.")

    if len(x.shape) == 4:  # this is 2D
        b, c2, h, w = x.shape  # c2 means c*2
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1)

    elif len(x.shape) == 5:  # this is 3D
        b, c2, h, w, d = x.shape  # c2 means c*2
        c = c2 // 2
        return x.view(b, 2, c, h, w, d).permute(0, 2, 3, 4, 5, 1)

    else:
        raise ValueError(f"only 2D (B,C*2,H,W) and 3D (B,C*2,H,W,D) data are supported but x has shape {x.shape}")


def reshape_channel_to_batch_dim(x: Tensor) -> Tuple[Tensor, int]:
    """
    Combines batch and channel dimensions.

    Args:
        x: input of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data

    Returns:
        A tuple containing:
            (1) output of shape (B*C,1,...)
            (2) batch size
    """

    if len(x.shape) == 5:  # this is 2D
        b, c, h, w, two = x.shape
        return x.contiguous().view(b * c, 1, h, w, two), b

    elif len(x.shape) == 6:  # this is 3D
        b, c, h, w, d, two = x.shape
        return x.contiguous().view(b * c, 1, h, w, d, two), b

    else:
        raise ValueError(f"only 2D (B,C,H,W,2) and 3D (B,C,H,W,D,2) data are supported but x has shape {x.shape}")


def reshape_batch_channel_to_channel_dim(x: Tensor, batch_size: int) -> Tensor:
    """
    Detaches batch and channel dimensions.

    Args:
        x: input of shape (B*C,1,H,W,2) for 2D data or (B*C,1,H,W,D,2) for 3D data
        batch_size: batch size

    Returns:
        output of shape (B,C,...)
    """
    if len(x.shape) == 5:  # this is 2D
        bc, one, h, w, two = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, h, w, two)

    elif len(x.shape) == 6:  # this is 3D
        bc, one, h, w, d, two = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, h, w, d, two)

    else:
        raise ValueError(f"only 2D (B*C,1,H,W,2) and 3D (B*C,1,H,W,D,2) data are supported but x has shape {x.shape}")


def complex_normalize(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
    """
    Performs layer mean-std normalization for complex data. Normalization is done for each batch member
    along each part (part refers to real and imaginary parts), separately.

    Args:
        x: input of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data

    Returns:
        A tuple containing
            (1) normalized output of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
            (2) mean
            (3) std
    """
    if len(x.shape) == 4:  # this is 2D
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        std = x.std(dim=2, unbiased=False).view(b, 2, 1, 1, 1).expand(b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    elif len(x.shape) == 5:  # this is 3D
        b, c, h, w, d = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w * d)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1, 1).expand(b, 2, c // 2, 1, 1, 1).contiguous().view(b, c, 1, 1, 1)
        std = (
            x.std(dim=2, unbiased=False)
            .view(b, 2, 1, 1, 1, 1)
            .expand(b, 2, c // 2, 1, 1, 1)
            .contiguous()
            .view(b, c, 1, 1, 1)
        )
        x = x.view(b, c, h, w, d)
        return (x - mean) / std, mean, std  # type: ignore

    else:
        raise ValueError(f"only 2D (B,C,H,W) and 3D (B,C,H,W,D) data are supported but x has shape {x.shape}")


def divisible_pad_t(
    x: Tensor, k: int = 16
) -> Tuple[Tensor, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], int, int, int]]:  # type: ignore
    """
    Pad input to feed into the network (torch script compatible)

    Args:
        x: input of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
        k: padding factor. each padded dimension will be divisible by k.

    Returns:
        A tuple containing
            (1) padded input
            (2) pad sizes (in order to reverse padding if needed)

    Example:
        .. code-block:: python

            import torch

            # 2D data
            x = torch.ones([3,2,50,70])
            x_pad,padding_sizes = divisible_pad_t(x, k=16)
            # the following line should print (3, 2, 64, 80)
            print(x_pad.shape)

            # 3D data
            x = torch.ones([3,2,50,70,80])
            x_pad,padding_sizes = divisible_pad_t(x, k=16)
            # the following line should print (3, 2, 64, 80, 80)
            print(x_pad.shape)

    """
    if len(x.shape) == 4:  # this is 2D
        b, c, h, w = x.shape
        w_mult = ((w - 1) | (k - 1)) + 1  # OR with (k-1) and then +1 makes sure padding is divisible by k
        h_mult = ((h - 1) | (k - 1)) + 1
        w_pad = floor_ceil((w_mult - w) / 2)
        h_pad = floor_ceil((h_mult - h) / 2)
        x = F.pad(x, w_pad + h_pad)
        # dummy values for the 3rd spatial dimension
        d_mult = -1
        d_pad = (-1, -1)
        pad_sizes = (h_pad, w_pad, d_pad, h_mult, w_mult, d_mult)

    elif len(x.shape) == 5:  # this is 3D
        b, c, h, w, d = x.shape
        w_mult = ((w - 1) | (k - 1)) + 1
        h_mult = ((h - 1) | (k - 1)) + 1
        d_mult = ((d - 1) | (k - 1)) + 1
        w_pad = floor_ceil((w_mult - w) / 2)
        h_pad = floor_ceil((h_mult - h) / 2)
        d_pad = floor_ceil((d_mult - d) / 2)
        x = F.pad(x, d_pad + w_pad + h_pad)
        pad_sizes = (h_pad, w_pad, d_pad, h_mult, w_mult, d_mult)

    else:
        raise ValueError(f"only 2D (B,C,H,W) and 3D (B,C,H,W,D) data are supported but x has shape {x.shape}")

    return x, pad_sizes


def inverse_divisible_pad_t(
    x: Tensor, pad_sizes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], int, int, int]
) -> Tensor:  # type: ignore
    """
    De-pad network output to match its original shape

    Args:
        x: input of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
        pad_sizes: padding values

    Returns:
        de-padded input
    """
    h_pad, w_pad, d_pad, h_mult, w_mult, d_mult = pad_sizes

    if len(x.shape) == 4:  # this is 2D
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    elif len(x.shape) == 5:  # this is 3D
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1], d_pad[0] : d_mult - d_pad[1]]

    else:
        raise ValueError(f"only 2D (B,C,H,W) and 3D (B,C,H,W,D) data are supported but x has shape {x.shape}")


def floor_ceil(n: float) -> Tuple[int, int]:
    """
    Returns floor and ceil of the input

    Args:
        n: input number

    Returns:
        A tuple containing:
            (1) floor(n)
            (2) ceil(n)
    """
    return math.floor(n), math.ceil(n)


def sensitivity_map_reduce(kspace: Tensor, sens_maps: Tensor, spatial_dims: int = 2) -> Tensor:
    """
    Reduces coil measurements to a corresponding image based on the given sens_maps. Let's say there
    are C coil measurements inside kspace, then this function multiplies the conjugate of each coil sensitivity map with the
    corresponding coil image. The result of this process will be C images. Summing those images together gives the
    resulting "reduced image."

    Args:
        kspace: 2D kspace (B,C,H,W,2) with the last dimension being 2 (for real/imaginary parts) and C denoting the
            coil dimension. 3D data will have the shape (B,C,H,W,D,2).
        sens_maps: sensitivity maps of the same shape as input x.
        spatial_dims: is 2 for 2D data and is 3 for 3D data

    Returns:
        reduction of x to (B,1,H,W,2) for 2D data or (B,1,H,W,D,2) for 3D data.
    """
    img = ifftn_centered_t(kspace, spatial_dims=spatial_dims, is_complex=True)  # inverse fourier transform
    return complex_mul_t(img, complex_conj_t(sens_maps)).sum(dim=1, keepdim=True)  # type: ignore


def sensitivity_map_expand(img: Tensor, sens_maps: Tensor, spatial_dims: int = 2) -> Tensor:
    """
    Expands an image to its corresponding coil images based on the given sens_maps. Let's say there
    are C coils. This function multiples image img with each coil sensitivity map in sens_maps and stacks
    the resulting C coil images along the channel dimension which is reserved for coils.

    Args:
        img: 2D image (B,1,H,W,2) with the last dimension being 2 (for real/imaginary parts). 3D data will have
            the shape (B,1,H,W,D,2).
        sens_maps: Sensitivity maps for combining coil images. The shape is (B,C,H,W,2) for 2D data
            or (B,C,H,W,D,2) for 3D data (C denotes the coil dimension).
        spatial_dims: is 2 for 2D data and is 3 for 3D data

    Returns:
        Expansion of x to (B,C,H,W,2) for 2D data and (B,C,H,W,D,2) for 3D data. The output is transferred
            to the frquency domain to yield coil measurements.
    """
    return fftn_centered_t(complex_mul_t(img, sens_maps), spatial_dims=spatial_dims, is_complex=True)  # type: ignore
