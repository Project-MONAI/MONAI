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


def gaussian_1d(sigma: Union[float, torch.Tensor], truncated: float = 4.0, approx: str = "simple"):
    """
    one dimensional Gaussian kernel.

    Args:
        sigma: std of the kernel
        truncated: tail length
        approx: Discrete Gaussian kernel type, available options are "simple" and "refined".
            The "refined" approximation corresponds to
            https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
            based on the modified Bessel functions.

    Raises:
        ValueError: When ``sigma`` is non-positive.

    Returns:
        1D torch tensor

    """
    sigma = torch.as_tensor(sigma, dtype=torch.float)
    if sigma <= 0.0 or truncated <= 0.0:
        raise ValueError(f"sigma and truncated must be positive, got {sigma} and {truncated}.")
    tail = int(sigma * truncated + 0.5)
    if approx.lower() == "simple":
        x = torch.arange(-tail, tail + 1, dtype=torch.float)
        t = 1.0 / (torch.tensor(2.0).sqrt() * sigma)
        out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
        return out.clamp(min=0)
    if approx.lower() == "refined":
        out = [_modified_bessel_0(sigma), _modified_bessel_1(sigma)]
        while len(out) <= tail:
            out.append(_modified_bessel_i(len(out), sigma))
        ans = out[:0:-1]
        ans.extend(out)
        ans = torch.stack(ans) * torch.exp(-sigma)
        ans /= ans.sum()
        return ans
    raise NotImplementedError(f"Unsupported option: approx='{approx}'.")


def _modified_bessel_0(x: Union[float, torch.Tensor]) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float)
    if torch.abs(x) < 3.75:
        y = (x / 3.75) * (x / 3.75)
        return 1.0 + y * (
            3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2))))
        )
    ax = torch.abs(x)
    y = 3.75 / ax
    ans = 0.916281e-2 + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2)))
    return (torch.exp(ax) / torch.sqrt(ax)) * (
        0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * ans)))
    )


def _modified_bessel_1(x: Union[float, torch.Tensor]) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float)
    if torch.abs(x) < 3.75:
        y = (x / 3.75) * (x / 3.75)
        ans = 0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3)))
        return torch.abs(x) * (0.5 + y * (0.87890594 + y * ans))
    ax = torch.abs(x)
    y = 3.75 / ax
    ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2))
    ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))))
    ans = ans * torch.exp(ax) / torch.sqrt(ax)
    return -ans if x < torch.tensor(0.0) else ans


def _modified_bessel_i(n: int, x: Union[float, torch.Tensor]) -> torch.Tensor:
    if n < 2:
        raise ValueError(f"n must be greater than 1, got n={n}.")
    x = torch.as_tensor(x, dtype=torch.float)
    if x == torch.tensor(0.0):
        return x
    tox = 2.0 / torch.abs(x)
    ans, bip, bi = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)
    m = int(2 * (n + np.floor(np.sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim
        if abs(bi) > 1.0e10:
            ans *= 1.0e-10
            bi *= 1.0e-10
            bip *= 1.0e-10
        if j == n:
            ans = bip
    ans *= _modified_bessel_0(x) / bi
    return -ans if x < torch.tensor(0.0) and (n % 2) == 1 else ans
