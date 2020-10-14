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

from typing import List, Optional, Sequence, Tuple, Union

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


def gaussian_1d(sigma: torch.Tensor, truncated: float = 4.0, approx: str = "erf", normalize: bool = False):
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
        ValueError: When ``sigma`` is non-positive.

    Returns:
        1D torch tensor

    """
    sigma = torch.as_tensor(sigma, dtype=torch.float, device=sigma.device if torch.is_tensor(sigma) else None)
    device = sigma.device
    if sigma <= 0.0 or truncated <= 0.0:
        raise ValueError(f"sigma and truncated must be positive, got {sigma} and {truncated}.")
    tail = int(sigma * truncated + 0.5)
    if approx.lower() == "erf":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
        t = 1.0 / (torch.tensor(2.0, device=device).sqrt() * sigma)
        out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
        out = out.clamp(min=0)
    elif approx.lower() == "sampled":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=sigma.device)
        sigma2 = sigma * sigma
        out = torch.exp(-0.5 / sigma2 * x ** 2)
        if not normalize:  # compute the normalizer
            out = out / (torch.tensor(2.0 * np.pi, device=device).sqrt() * sigma)
    elif approx.lower() == "scalespace":
        sigma2 = sigma * sigma
        out_pos: List[Optional[torch.Tensor]] = [None] * (tail + 1)
        out_pos[0] = _modified_bessel_0(sigma2)
        if len(out_pos) > 1:
            out_pos[1] = _modified_bessel_1(sigma2)
        for k in range(2, len(out_pos)):
            out_pos[k] = _modified_bessel_i(k, sigma2)
        out = out_pos[:0:-1]
        out.extend(out_pos)
        out = torch.stack(out) * torch.exp(-sigma2)
    else:
        raise NotImplementedError(f"Unsupported option: approx='{approx}'.")
    return out / out.sum() if normalize else out


def _modified_bessel_0(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if torch.is_tensor(x) else None)
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


def _modified_bessel_1(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if torch.is_tensor(x) else None)
    if torch.abs(x) < 3.75:
        y = (x / 3.75) * (x / 3.75)
        ans = 0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3)))
        return torch.abs(x) * (0.5 + y * (0.87890594 + y * ans))
    ax = torch.abs(x)
    y = 3.75 / ax
    ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2))
    ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))))
    ans = ans * torch.exp(ax) / torch.sqrt(ax)
    return -ans if x < 0.0 else ans


def _modified_bessel_i(n: int, x: torch.Tensor) -> torch.Tensor:
    if n < 2:
        raise ValueError(f"n must be greater than 1, got n={n}.")
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if torch.is_tensor(x) else None)
    device = x.device
    if x == 0.0:
        return x
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
