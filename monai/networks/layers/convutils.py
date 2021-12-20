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

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = ["same_padding", "stride_minus_kernel_padding", "calculate_out_shape", "gaussian_1d", "polyval"]


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
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)

    out_padding_np = stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


def calculate_out_shape(
    in_shape: Union[Sequence[int], int, np.ndarray],
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


def gaussian_1d(
    sigma: torch.Tensor, truncated: float = 4.0, approx: str = "erf", normalize: bool = False
) -> torch.Tensor:
    """
    one dimensional Gaussian kernel.

    Args:
        sigma: std of the kernel
        truncated: tail length
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".

            - ``erf`` approximation interpolates the error function;
            - ``sampled`` uses a sampled Gaussian kernel;
            - ``scalespace`` corresponds to
              https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
              based on the modified Bessel functions.

        normalize: whether to normalize the kernel with `kernel.sum()`.

    Raises:
        ValueError: When ``truncated`` is non-positive.

    Returns:
        1D torch tensor

    """
    sigma = torch.as_tensor(sigma, dtype=torch.float, device=sigma.device if isinstance(sigma, torch.Tensor) else None)
    device = sigma.device
    if truncated <= 0.0:
        raise ValueError(f"truncated must be positive, got {truncated}.")
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    if approx.lower() == "erf":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
        t = 0.70710678 / torch.abs(sigma)
        out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
        out = out.clamp(min=0)
    elif approx.lower() == "sampled":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=sigma.device)
        out = torch.exp(-0.5 / (sigma * sigma) * x ** 2)
        if not normalize:  # compute the normalizer
            out = out / (2.5066282 * sigma)
    elif approx.lower() == "scalespace":
        sigma2 = sigma * sigma
        out_pos: List[Optional[torch.Tensor]] = [None] * (tail + 1)
        out_pos[0] = _modified_bessel_0(sigma2)
        out_pos[1] = _modified_bessel_1(sigma2)
        for k in range(2, len(out_pos)):
            out_pos[k] = _modified_bessel_i(k, sigma2)
        out = out_pos[:0:-1]
        out.extend(out_pos)
        out = torch.stack(out) * torch.exp(-sigma2)
    else:
        raise NotImplementedError(f"Unsupported option: approx='{approx}'.")
    return out / out.sum() if normalize else out  # type: ignore


def polyval(coef, x) -> torch.Tensor:
    """
    Evaluates the polynomial defined by `coef` at `x`.

    For a 1D sequence of coef (length n), evaluate::

        y = coef[n-1] + x * (coef[n-2] + ... + x * (coef[1] + x * coef[0]))

    Args:
        coef: a sequence of floats representing the coefficients of the polynomial
        x: float or a sequence of floats representing the variable of the polynomial

    Returns:
        1D torch tensor
    """
    device = x.device if isinstance(x, torch.Tensor) else None
    coef = torch.as_tensor(coef, dtype=torch.float, device=device)
    if coef.ndim == 0 or (len(coef) < 1):
        return torch.zeros(x.shape)
    x = torch.as_tensor(x, dtype=torch.float, device=device)
    ans = coef[0]
    for c in coef[1:]:
        ans = ans * x + c
    return ans  # type: ignore


def _modified_bessel_0(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if isinstance(x, torch.Tensor) else None)
    if torch.abs(x) < 3.75:
        y = x * x / 14.0625
        return polyval([0.45813e-2, 0.360768e-1, 0.2659732, 1.2067492, 3.0899424, 3.5156229, 1.0], y)
    ax = torch.abs(x)
    y = 3.75 / ax
    _coef = [
        0.392377e-2,
        -0.1647633e-1,
        0.2635537e-1,
        -0.2057706e-1,
        0.916281e-2,
        -0.157565e-2,
        0.225319e-2,
        0.1328592e-1,
        0.39894228,
    ]
    return polyval(_coef, y) * torch.exp(ax) / torch.sqrt(ax)


def _modified_bessel_1(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if isinstance(x, torch.Tensor) else None)
    if torch.abs(x) < 3.75:
        y = x * x / 14.0625
        _coef = [0.32411e-3, 0.301532e-2, 0.2658733e-1, 0.15084934, 0.51498869, 0.87890594, 0.5]
        return torch.abs(x) * polyval(_coef, y)
    ax = torch.abs(x)
    y = 3.75 / ax
    _coef = [
        -0.420059e-2,
        0.1787654e-1,
        -0.2895312e-1,
        0.2282967e-1,
        -0.1031555e-1,
        0.163801e-2,
        -0.362018e-2,
        -0.3988024e-1,
        0.39894228,
    ]
    ans = polyval(_coef, y) * torch.exp(ax) / torch.sqrt(ax)
    return -ans if x < 0.0 else ans


def _modified_bessel_i(n: int, x: torch.Tensor) -> torch.Tensor:
    if n < 2:
        raise ValueError(f"n must be greater than 1, got n={n}.")
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if isinstance(x, torch.Tensor) else None)
    if x == 0.0:
        return x
    device = x.device
    tox = 2.0 / torch.abs(x)
    ans, bip, bi = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    m = int(2 * (n + np.floor(np.sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim
        if abs(bi) > 1.0e10:
            ans = ans * 1.0e-10
            bi = bi * 1.0e-10
            bip = bip * 1.0e-10
        if j == n:
            ans = bip
    ans = ans * _modified_bessel_0(x) / bi
    return -ans if x < 0.0 and (n % 2) == 1 else ans
