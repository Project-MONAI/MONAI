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

from typing import Sequence

from torch import Tensor


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


def reshape_channel_to_batch_dim(x: Tensor) -> Sequence:
    """
    Combines batch and channel dimensions.

    Args:
        x: Ndim input of shape shape (B,C,...)

    Returns:
        A tuple containing:
            (1) output of of shape (B*C,1,...)
            (2) batch size
    """
    b, c, *other = x.shape
    return x.contiguous().view(b * c, 1, *other), b


def reshape_batch_channel_to_channel_dim(x: Tensor, batch_size: int) -> Tensor:
    """
    Detaches batch and channel dimensions.

    Args:
        x: Ndim input of shape (B*C,1,...)
        batch_size: batch size

    Returns:
        output of shape (B,C,...)
    """
    bc, one, *other = x.shape  # bc represents B*C
    c = bc // batch_size
    return x.view(batch_size, c, *other)


def complex_normalize(x: Tensor) -> Sequence:  # type: ignore
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
