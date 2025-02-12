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

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.utils.misc import is_module_ver_at_least
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

__all__ = [
    "allclose",
    "moveaxis",
    "in1d",
    "clip",
    "percentile",
    "where",
    "argwhere",
    "argsort",
    "nonzero",
    "floor_divide",
    "unravel_index",
    "unravel_indices",
    "ravel",
    "any_np_pt",
    "maximum",
    "concatenate",
    "cumsum",
    "isfinite",
    "searchsorted",
    "repeat",
    "isnan",
    "ascontiguousarray",
    "stack",
    "mode",
    "unique",
    "max",
    "min",
    "median",
    "mean",
    "std",
    "softplus",
]


def softplus(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """stable softplus through `np.logaddexp` with equivalent implementation for torch.

    Args:
        x: array/tensor.

    Returns:
        Softplus of the input.
    """
    if isinstance(x, np.ndarray):
        return np.logaddexp(np.zeros_like(x), x)
    return torch.logaddexp(torch.zeros_like(x), x)


def allclose(a: NdarrayTensor, b: NdarrayOrTensor, rtol=1e-5, atol=1e-8, equal_nan=False) -> bool:
    """`np.allclose` with equivalent implementation for torch."""
    b, *_ = convert_to_dst_type(b, a, wrap_sequence=True)
    if isinstance(a, np.ndarray):
        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)  # type: ignore


def moveaxis(x: NdarrayOrTensor, src: int | Sequence[int], dst: int | Sequence[int]) -> NdarrayOrTensor:
    """`moveaxis` for pytorch and numpy"""
    if isinstance(x, torch.Tensor):
        return torch.movedim(x, src, dst)  # type: ignore
    return np.moveaxis(x, src, dst)


def in1d(x, y):
    """`np.in1d` with equivalent implementation for torch."""
    if isinstance(x, np.ndarray):
        return np.isin(x, y)
    return (x[..., None] == torch.tensor(y, device=x.device)).any(-1).view(-1)


def clip(a: NdarrayOrTensor, a_min, a_max) -> NdarrayOrTensor:
    """`np.clip` with equivalent implementation for torch."""
    result: NdarrayOrTensor
    if isinstance(a, np.ndarray):
        result = np.clip(a, a_min, a_max)
    else:
        result = torch.clamp(a, a_min, a_max)
    return result


def percentile(
    x: NdarrayOrTensor, q, dim: int | None = None, keepdim: bool = False, **kwargs
) -> NdarrayOrTensor | float | int:
    """`np.percentile` with equivalent implementation for torch.

    Pytorch uses `quantile`. For more details please refer to:
    https://pytorch.org/docs/stable/generated/torch.quantile.html.
    https://numpy.org/doc/stable/reference/generated/numpy.percentile.html.

    Args:
        x: input data.
        q: percentile to compute (should in range 0 <= q <= 100).
        dim: the dim along which the percentiles are computed. default is to compute the percentile
            along a flattened version of the array.
        keepdim: whether the output data has dim retained or not.
        kwargs: if `x` is numpy array, additional args for `np.percentile`, more details:
            https://numpy.org/doc/stable/reference/generated/numpy.percentile.html.

    Returns:
        Resulting value (scalar)
    """
    q_np = convert_data_type(q, output_type=np.ndarray, wrap_sequence=True)[0]
    if ((q_np < 0) | (q_np > 100)).any():
        raise ValueError(f"q values must be in [0, 100], got values: {q}.")
    result: NdarrayOrTensor | float | int
    if isinstance(x, np.ndarray) or (isinstance(x, torch.Tensor) and torch.numel(x) > 1_000_000):  # pytorch#64947
        _x = convert_data_type(x, output_type=np.ndarray)[0]
        result = np.percentile(_x, q_np, axis=dim, keepdims=keepdim, **kwargs)
        result = convert_to_dst_type(result, x)[0]
    else:
        q = convert_to_dst_type(q_np / 100.0, x)[0]
        result = torch.quantile(x, q, dim=dim, keepdim=keepdim)
    return result


def where(condition: NdarrayOrTensor, x=None, y=None) -> NdarrayOrTensor:
    """
    Note that `torch.where` may convert y.dtype to x.dtype.
    """
    result: NdarrayOrTensor
    if isinstance(condition, np.ndarray):
        if x is not None:
            result = np.where(condition, x, y)
        else:
            result = np.where(condition)  # type: ignore
    else:
        if x is not None:
            x = torch.as_tensor(x, device=condition.device)
            y = torch.as_tensor(y, device=condition.device, dtype=x.dtype)
            result = torch.where(condition, x, y)
        else:
            result = torch.where(condition)  # type: ignore
    return result


def argwhere(a: NdarrayTensor) -> NdarrayTensor:
    """`np.argwhere` with equivalent implementation for torch.

    Args:
        a: input data.

    Returns:
        Indices of elements that are non-zero. Indices are grouped by element.
        This array will have shape (N, a.ndim) where N is the number of non-zero items.
    """
    if isinstance(a, np.ndarray):
        return np.argwhere(a)  # type: ignore
    return torch.argwhere(a)  # type: ignore


def argsort(a: NdarrayTensor, axis: int | None = -1) -> NdarrayTensor:
    """`np.argsort` with equivalent implementation for torch.

    Args:
        a: the array/tensor to sort.
        axis: axis along which to sort.

    Returns:
        Array/Tensor of indices that sort a along the specified axis.
    """
    if isinstance(a, np.ndarray):
        return np.argsort(a, axis=axis)  # type: ignore
    return torch.argsort(a, dim=axis)  # type: ignore


def nonzero(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.nonzero` with equivalent implementation for torch.

    Args:
        x: array/tensor.

    Returns:
        Index unravelled for given shape
    """
    if isinstance(x, np.ndarray):
        return np.nonzero(x)[0]
    return torch.nonzero(x).flatten()


def floor_divide(a: NdarrayOrTensor, b) -> NdarrayOrTensor:
    """`np.floor_divide` with equivalent implementation for torch.

    As of pt1.8, use `torch.div(..., rounding_mode="floor")`, and
    before that, use `torch.floor_divide`.

    Args:
        a: first array/tensor
        b: scalar to divide by

    Returns:
        Element-wise floor division between two arrays/tensors.
    """
    if isinstance(a, torch.Tensor):
        if is_module_ver_at_least(torch, (1, 8, 0)):
            return torch.div(a, b, rounding_mode="floor")
        return torch.floor_divide(a, b)
    return np.floor_divide(a, b)


def unravel_index(idx, shape) -> NdarrayOrTensor:
    """`np.unravel_index` with equivalent implementation for torch.

    Args:
        idx: index to unravel.
        shape: shape of array/tensor.

    Returns:
        Index unravelled for given shape
    """
    if isinstance(idx, torch.Tensor):
        coord = []
        for dim in reversed(shape):
            coord.append(idx % dim)
            idx = floor_divide(idx, dim)
        return torch.stack(coord[::-1])
    return np.asarray(np.unravel_index(idx, shape))


def unravel_indices(idx, shape) -> NdarrayOrTensor:
    """Computing unravel coordinates from indices.

    Args:
        idx: a sequence of indices to unravel.
        shape: shape of array/tensor.

    Returns:
        Stacked indices unravelled for given shape
    """
    lib_stack = torch.stack if isinstance(idx[0], torch.Tensor) else np.stack
    return lib_stack([unravel_index(i, shape) for i in idx])  # type: ignore


def ravel(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.ravel` with equivalent implementation for torch.

    Args:
        x: array/tensor to ravel.

    Returns:
        Return a contiguous flattened array/tensor.
    """
    if isinstance(x, torch.Tensor):
        if hasattr(torch, "ravel"):  # `ravel` is new in torch 1.8.0
            return x.ravel()
        return x.flatten().contiguous()
    return np.ravel(x)


def any_np_pt(x: NdarrayOrTensor, axis: int | Sequence[int]) -> NdarrayOrTensor:
    """`np.any` with equivalent implementation for torch.

    For pytorch, convert to boolean for compatibility with older versions.

    Args:
        x: input array/tensor.
        axis: axis to perform `any` over.

    Returns:
        Return a contiguous flattened array/tensor.
    """
    if isinstance(x, np.ndarray):
        return np.any(x, axis)  # type: ignore

    # pytorch can't handle multiple dimensions to `any` so loop across them
    axis = [axis] if not isinstance(axis, Sequence) else axis
    for ax in axis:
        try:
            x = torch.any(x, ax)
        except RuntimeError:
            # older versions of pytorch require the input to be cast to boolean
            x = torch.any(x.bool(), ax)
    return x


def maximum(a: NdarrayOrTensor, b: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.maximum` with equivalent implementation for torch.

    Args:
        a: first array/tensor.
        b: second array/tensor.

    Returns:
        Element-wise maximum between two arrays/tensors.
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.maximum(a, b)
    return np.maximum(a, b)


def concatenate(to_cat: Sequence[NdarrayOrTensor], axis: int = 0, out=None) -> NdarrayOrTensor:
    """`np.concatenate` with equivalent implementation for torch (`torch.cat`)."""
    if isinstance(to_cat[0], np.ndarray):
        return np.concatenate(to_cat, axis, out)  # type: ignore
    return torch.cat(to_cat, dim=axis, out=out)  # type: ignore


def cumsum(a: NdarrayOrTensor, axis=None, **kwargs) -> NdarrayOrTensor:
    """
    `np.cumsum` with equivalent implementation for torch.

    Args:
        a: input data to compute cumsum.
        axis: expected axis to compute cumsum.
        kwargs: if `a` is PyTorch Tensor, additional args for `torch.cumsum`, more details:
            https://pytorch.org/docs/stable/generated/torch.cumsum.html.

    """

    if isinstance(a, np.ndarray):
        return np.cumsum(a, axis)  # type: ignore
    if axis is None:
        return torch.cumsum(a[:], 0, **kwargs)
    return torch.cumsum(a, dim=axis, **kwargs)


def isfinite(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.isfinite` with equivalent implementation for torch."""
    if not isinstance(x, torch.Tensor):
        return np.isfinite(x)  # type: ignore
    return torch.isfinite(x)


def searchsorted(a: NdarrayTensor, v: NdarrayOrTensor, right=False, sorter=None, **kwargs) -> NdarrayTensor:
    """
    `np.searchsorted` with equivalent implementation for torch.

    Args:
        a: numpy array or tensor, containing monotonically increasing sequence on the innermost dimension.
        v: containing the search values.
        right: if False, return the first suitable location that is found, if True, return the last such index.
        sorter: if `a` is numpy array, optional array of integer indices that sort array `a` into ascending order.
        kwargs: if `a` is PyTorch Tensor, additional args for `torch.searchsorted`, more details:
            https://pytorch.org/docs/stable/generated/torch.searchsorted.html.

    """
    side = "right" if right else "left"
    if isinstance(a, np.ndarray):
        return np.searchsorted(a, v, side, sorter)  # type: ignore
    return torch.searchsorted(a, v, right=right, **kwargs)  # type: ignore


def repeat(a: NdarrayOrTensor, repeats: int, axis: int | None = None, **kwargs) -> NdarrayOrTensor:
    """
    `np.repeat` with equivalent implementation for torch (`repeat_interleave`).

    Args:
        a: input data to repeat.
        repeats: number of repetitions for each element, repeats is broadcast to fit the shape of the given axis.
        axis: axis along which to repeat values.
        kwargs: if `a` is PyTorch Tensor, additional args for `torch.repeat_interleave`, more details:
            https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html.

    """
    if isinstance(a, np.ndarray):
        return np.repeat(a, repeats, axis)
    return torch.repeat_interleave(a, repeats, dim=axis, **kwargs)


def isnan(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.isnan` with equivalent implementation for torch.

    Args:
        x: array/tensor.

    """
    if isinstance(x, np.ndarray):
        return np.isnan(x)  # type: ignore
    return torch.isnan(x)


T = TypeVar("T")


def ascontiguousarray(x: NdarrayTensor | T, **kwargs) -> NdarrayOrTensor | T:
    """`np.ascontiguousarray` with equivalent implementation for torch (`contiguous`).

    Args:
        x: array/tensor.
        kwargs: if `x` is PyTorch Tensor, additional args for `torch.contiguous`, more details:
            https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html.

    """
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x
        return np.ascontiguousarray(x)
    if isinstance(x, torch.Tensor):
        return x.contiguous(**kwargs)
    return x


def stack(x: Sequence[NdarrayTensor], dim: int) -> NdarrayTensor:
    """`np.stack` with equivalent implementation for torch.

    Args:
        x: array/tensor.
        dim: dimension along which to perform the stack (referred to as `axis` by numpy).
    """
    if isinstance(x[0], np.ndarray):
        return np.stack(x, dim)  # type: ignore
    return torch.stack(x, dim)  # type: ignore


def mode(x: NdarrayTensor, dim: int = -1, to_long: bool = True) -> NdarrayTensor:
    """`torch.mode` with equivalent implementation for numpy.

    Args:
        x: array/tensor.
        dim: dimension along which to perform `mode` (referred to as `axis` by numpy).
        to_long: convert input to long before performing mode.
    """
    dtype = torch.int64 if to_long else None
    x_t, *_ = convert_data_type(x, torch.Tensor, dtype=dtype)
    o_t = torch.mode(x_t, dim).values
    o, *_ = convert_to_dst_type(o_t, x)
    return o


def unique(x: NdarrayTensor, **kwargs) -> NdarrayTensor:
    """`torch.unique` with equivalent implementation for numpy.

    Args:
        x: array/tensor.
    """
    return np.unique(x, **kwargs) if isinstance(x, (np.ndarray, list)) else torch.unique(x, **kwargs)  # type: ignore


def linalg_inv(x: NdarrayTensor) -> NdarrayTensor:
    """`torch.linalg.inv` with equivalent implementation for numpy.

    Args:
        x: array/tensor.
    """
    if isinstance(x, torch.Tensor) and hasattr(torch, "inverse"):  # pytorch 1.7.0
        return torch.inverse(x)  # type: ignore
    return torch.linalg.inv(x) if isinstance(x, torch.Tensor) else np.linalg.inv(x)  # type: ignore


def max(x: NdarrayTensor, dim: int | tuple | None = None, **kwargs) -> NdarrayTensor:
    """`torch.max` with equivalent implementation for numpy

    Args:
        x: array/tensor.

    Returns:
        the maximum of x.

    """

    ret: NdarrayTensor
    if dim is None:
        ret = np.max(x, **kwargs) if isinstance(x, (np.ndarray, list)) else torch.max(x, **kwargs)  # type: ignore
    else:
        if isinstance(x, (np.ndarray, list)):
            ret = np.max(x, axis=dim, **kwargs)
        else:
            ret = torch.max(x, int(dim), **kwargs)  # type: ignore

    return ret[0] if isinstance(ret, tuple) else ret


def mean(x: NdarrayTensor, dim: int | tuple | None = None, **kwargs) -> NdarrayTensor:
    """`torch.mean` with equivalent implementation for numpy

    Args:
        x: array/tensor.

    Returns:
        the mean of x
    """

    ret: NdarrayTensor
    if dim is None:
        ret = np.mean(x, **kwargs) if isinstance(x, (np.ndarray, list)) else torch.mean(x, **kwargs)  # type: ignore
    else:
        if isinstance(x, (np.ndarray, list)):
            ret = np.mean(x, axis=dim, **kwargs)
        else:
            ret = torch.mean(x, int(dim), **kwargs)  # type: ignore

    return ret


def median(x: NdarrayTensor, dim: int | tuple | None = None, **kwargs) -> NdarrayTensor:
    """`torch.median` with equivalent implementation for numpy

    Args:
        x: array/tensor.

    Returns
        the median of x.
    """

    ret: NdarrayTensor
    if dim is None:
        ret = np.median(x, **kwargs) if isinstance(x, (np.ndarray, list)) else torch.median(x, **kwargs)  # type: ignore
    else:
        if isinstance(x, (np.ndarray, list)):
            ret = np.median(x, axis=dim, **kwargs)
        else:
            ret = torch.median(x, int(dim), **kwargs)  # type: ignore

    return ret


def min(x: NdarrayTensor, dim: int | tuple | None = None, **kwargs) -> NdarrayTensor:
    """`torch.min` with equivalent implementation for numpy

    Args:
        x: array/tensor.

    Returns:
        the minimum of x.
    """

    ret: NdarrayTensor
    if dim is None:
        ret = np.min(x, **kwargs) if isinstance(x, (np.ndarray, list)) else torch.min(x, **kwargs)  # type: ignore
    else:
        if isinstance(x, (np.ndarray, list)):
            ret = np.min(x, axis=dim, **kwargs)
        else:
            ret = torch.min(x, int(dim), **kwargs)  # type: ignore

    return ret[0] if isinstance(ret, tuple) else ret


def std(x: NdarrayTensor, dim: int | tuple | None = None, unbiased: bool = False) -> NdarrayTensor:
    """`torch.std` with equivalent implementation for numpy

    Args:
        x: array/tensor.

    Returns:
        the standard deviation of x.
    """

    ret: NdarrayTensor
    if dim is None:
        ret = np.std(x) if isinstance(x, (np.ndarray, list)) else torch.std(x, unbiased)  # type: ignore
    else:
        if isinstance(x, (np.ndarray, list)):
            ret = np.std(x, axis=dim)
        else:
            ret = torch.std(x, int(dim), unbiased)  # type: ignore

    return ret


def sum(x: NdarrayTensor, dim: int | tuple | None = None, **kwargs) -> NdarrayTensor:
    """`torch.sum` with equivalent implementation for numpy

    Args:
        x: array/tensor.

    Returns:
        the sum of x.
    """

    ret: NdarrayTensor
    if dim is None:
        ret = np.sum(x, **kwargs) if isinstance(x, (np.ndarray, list)) else torch.sum(x, **kwargs)  # type: ignore
    else:
        if isinstance(x, (np.ndarray, list)):
            ret = np.sum(x, axis=dim, **kwargs)
        else:
            ret = torch.sum(x, int(dim), **kwargs)  # type: ignore

    return ret
