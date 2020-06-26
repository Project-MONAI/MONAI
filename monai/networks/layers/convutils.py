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

import numpy as np

__all__ = ["same_padding", "calculate_out_shape", "gaussian_1d"]


def same_padding(kernel_size, dilation=1):
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: same padding not available for k={kernel_size} and d={dilation}.

    """

    kernel_size = np.atleast_1d(kernel_size)
    dilation = np.atleast_1d(dilation)

    if np.any((kernel_size - 1) * dilation % 2 == 1):
        raise NotImplementedError(f"Same padding not available for k={kernel_size} and d={dilation}.")

    padding = (kernel_size - 1) / 2 * dilation
    padding = tuple(int(p) for p in padding)

    return tuple(padding) if len(padding) > 1 else padding[0]


def calculate_out_shape(in_shape, kernel_size, stride, padding):
    """
    Calculate the output tensor shape when applying a convolution to a tensor of shape `inShape` with kernel size
    `kernel_size`, stride value `stride`, and input padding value `padding`. All arguments can be scalars or multiple
    values, return value is a scalar if all inputs are scalars.
    """
    in_shape = np.atleast_1d(in_shape)
    out_shape = ((in_shape - kernel_size + padding + padding) // stride) + 1
    out_shape = tuple(int(s) for s in out_shape)

    return tuple(out_shape) if len(out_shape) > 1 else out_shape[0]


def gaussian_1d(sigma, truncated=4.0):
    """
    one dimensional gaussian kernel.

    Args:
        sigma: std of the kernel
        truncated: tail length

    Returns:
        1D numpy array

    Raises:
        ValueError: sigma must be positive

    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    tail = int(sigma * truncated + 0.5)
    sigma2 = sigma * sigma
    x = np.arange(-tail, tail + 1)
    out = np.exp(-0.5 / sigma2 * x ** 2)
    out /= out.sum()
    return out
