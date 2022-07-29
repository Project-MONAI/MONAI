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
from typing import Sequence

from torch import Tensor
from torch.nn import functional as F



def complex_to_channel_dim(self, x: Tensor) -> Tensor:
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


def channel_complex_to_last_dim(self, x: Tensor) -> Tensor:
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


def complex_normalize(self, x: Tensor) -> Sequence:
    """
    Performs group mean-std normalization for complex data. Normalization is done for each batch member
    and each part (real and imaginary parts), separately. To see what "group" means, mean of
    an input of shape (B,C,H,W) will be (B,).

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
        std = x.std(dim=2, unbiased=False).view(b, 2, 1, 1, 1, 1).expand(b, 2, c // 2, 1, 1, 1).contiguous().view(b, c, 1, 1, 1)
        x = x.view(b, c, h, w, d)
        return (x - mean) / std, mean, std


def reverse_complex_normalize(self, x: Tensor, mean: float, std: float) -> Tensor:
    """
    Reverses the normalization done by complex_normalize

    Args:
        x: input of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
        mean: mean before normalization
        std: std before normalization

    Returns:
        denormalized output of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
    """
    return x * std + mean


def floor_ceil(n: float) -> Sequence:
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


def pad(self, x: Tensor) -> Sequence:
    """
    Pad input to feed into the network

    Args:
        x: input of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data

    Returns:
        A tuple containing
            (1) padded input
            (2) pad sizes (in order to reverse padding if needed)
    """
    if len(x.shape) == 4:  # this is 2D
        b, c, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = floor_ceil((w_mult - w) / 2)
        h_pad = floor_ceil((h_mult - h) / 2)
        x = F.pad(x, w_pad + h_pad)  # type: ignore
        return x, (h_pad, w_pad, h_mult, w_mult)

    elif len(x.shape) == 5:  # this is 3D
        b, c, h, w, d = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        d_mult = ((d - 1) | 15) + 1
        w_pad = floor_ceil((w_mult - w) / 2)
        h_pad = floor_ceil((h_mult - h) / 2)
        d_pad = floor_ceil((d_mult - d) / 2)
        x = F.pad(x, w_pad + h_pad + d_pad)
        return x, (h_pad, w_pad, d_pad, h_mult, w_mult, d_mult)


def reverse_pad(self, x: Tensor, pad_sizes: Sequence) -> Tensor:
    """
    De-pad network output to match its original shape

    Args:
        x: input of shape (B,C,H,W) for 2D data or (B,C,H,W,D) for 3D data
        pad_sizes: padding values

    Returns:
        de-padded input
    """
    if len(x.shape) == 4:  # this is 2D
        h_pad, w_pad, h_mult, w_mult = pad_sizes
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    if len(x.shape) == 5:  # this is 3D
        h_pad, w_pad, d_pad, h_mult, w_mult, d_mult = pad_sizes
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1], d_pad[0] : d_mult - d_pad[1]]
