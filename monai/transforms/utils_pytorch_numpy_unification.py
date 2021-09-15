# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor

__all__ = [
    "moveaxis",
    "in1d",
    "clip",
    "percentile",
    "where",
]


def moveaxis(x: NdarrayOrTensor, src: int, dst: int) -> NdarrayOrTensor:
    """`moveaxis` for pytorch and numpy, using `permute` for pytorch ver < 1.8"""
    if isinstance(x, torch.Tensor):
        if hasattr(torch, "moveaxis"):
            return torch.moveaxis(x, src, dst)
        return _moveaxis_with_permute(x, src, dst)  # type: ignore
    if isinstance(x, np.ndarray):
        return np.moveaxis(x, src, dst)
    raise RuntimeError()


def _moveaxis_with_permute(x, src, dst):
    # get original indices
    indices = list(range(x.ndim))
    # make src and dst positive
    if src < 0:
        src = len(indices) + src
    if dst < 0:
        dst = len(indices) + dst
    # remove desired index and insert it in new position
    indices.pop(src)
    indices.insert(dst, src)
    return x.permute(indices)


def in1d(x, y):
    """`np.in1d` with equivalent implementation for torch."""
    if isinstance(x, np.ndarray):
        return np.in1d(x, y)
    return (x[..., None] == torch.tensor(y, device=x.device)).any(-1).view(-1)


def clip(a: NdarrayOrTensor, a_min, a_max) -> NdarrayOrTensor:
    """`np.clip` with equivalent implementation for torch."""
    result: NdarrayOrTensor
    if isinstance(a, np.ndarray):
        result = np.clip(a, a_min, a_max)
    else:
        result = torch.clip(a, a_min, a_max)
    return result


def percentile(x: NdarrayOrTensor, q) -> Union[NdarrayOrTensor, float, int]:
    """`np.percentile` with equivalent implementation for torch.

    Pytorch uses `quantile`, but this functionality is only available from v1.7.
    For earlier methods, we calculate it ourselves. This doesn't do interpolation,
    so is the equivalent of ``numpy.percentile(..., interpolation="nearest")``.

    Args:
        x: input data
        q: percentile to compute (should in range 0 <= q <= 100)

    Returns:
        Resulting value (scalar)
    """
    if np.isscalar(q):
        if not 0 <= q <= 100:
            raise ValueError
    else:
        if any(q < 0) or any(q > 100):
            raise ValueError
    result: Union[NdarrayOrTensor, float, int]
    if isinstance(x, np.ndarray):
        result = np.percentile(x, q)
    else:
        q = torch.tensor(q, device=x.device)
        if hasattr(torch, "quantile"):
            result = torch.quantile(x, q / 100.0)
        else:
            # Note that ``kthvalue()`` works one-based, i.e., the first sorted value
            # corresponds to k=1, not k=0. Thus, we need the `1 +`.
            k = 1 + (0.01 * q * (x.numel() - 1)).round().int()
            if k.numel() > 1:
                r = [x.view(-1).kthvalue(int(_k)).values.item() for _k in k]
                result = torch.tensor(r, device=x.device)
            else:
                result = x.view(-1).kthvalue(int(k)).values.item()

    return result


def where(condition: NdarrayOrTensor, x, y) -> NdarrayOrTensor:
    """
    Note that `torch.where` may convert y.dtype to x.dtype.
    """
    result: NdarrayOrTensor
    if isinstance(condition, np.ndarray):
        result = np.where(condition, x, y)
    else:
        x = torch.as_tensor(x, device=condition.device)
        y = torch.as_tensor(y, device=condition.device, dtype=x.dtype)
        result = torch.where(condition, x, y)
    return result
