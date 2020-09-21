# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Tuple, Union

import numpy as np
import torch

__all__ = ["same_padding", "stride_minus_kernel_padding", "calculate_out_shape", "gaussian_1d"]


def same_padding(
    kernel_size: Union[Sequence[int], int], dilation: Union[Sequence[int], int] = 1
) -> Union[Tuple[int, ...], int]:
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.

    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def stride_minus_kernel_padding(
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)

    out_padding_np = stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


def calculate_out_shape(
    in_shape: Union[Sequence[int], int],
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
) -> Union[Tuple[int, ...], int]:
    """
    Calculate the output tensor shape when applying a convolution to a tensor of shape `inShape` with kernel size
    `kernel_size`, stride value `stride`, and input padding value `padding`. All arguments can be scalars or multiple
    values, return value is a scalar if all inputs are scalars.
    """
    in_shape_np = np.atleast_1d(in_shape)
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_shape_np = ((in_shape_np - kernel_size_np + padding_np + padding_np) // stride_np) + 1
    out_shape = tuple(int(s) for s in out_shape_np)

    return out_shape if len(out_shape) > 1 else out_shape[0]


def gaussian_1d(sigma: Union[float, torch.Tensor], truncated: float = 4.0) -> np.ndarray:
    """
    one dimensional gaussian kernel.

    Args:
        sigma: std of the kernel
        truncated: tail length

    Raises:
        ValueError: When ``sigma`` is non-positive.

    Returns:
        1D torch tensor

    """
    sigma = torch.as_tensor(sigma).float()
    if sigma <= 0 or truncated <= 0:
        raise ValueError(f"sigma and truncated must be positive, got {sigma} and {truncated}.")
    tail = int(sigma * truncated + 0.5)
    x = torch.arange(-tail, tail + 1).float()
    t = 1 / (torch.tensor(2.0).sqrt() * sigma)
    out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    return out.clamp(min=0)
