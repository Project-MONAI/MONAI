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

from typing import Optional, Sequence, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.misc import ensure_tuple, is_module_ver_at_least

__all__ = [
    "moveaxis",
    "in1d",
    "clip",
    "percentile",
    "where",
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
]


def moveaxis(x: NdarrayOrTensor, src: Union[int, Sequence[int]], dst: Union[int, Sequence[int]]) -> NdarrayOrTensor:
    """`moveaxis` for pytorch and numpy, using `permute` for pytorch version < 1.7"""
    if isinstance(x, torch.Tensor):
        if hasattr(torch, "movedim"):  # `movedim` is new in torch 1.7.0
            # torch.moveaxis is a recent alias since torch 1.8.0
            return torch.movedim(x, src, dst)  # type: ignore
        return _moveaxis_with_permute(x, src, dst)  # type: ignore
    return np.moveaxis(x, src, dst)


def _moveaxis_with_permute(
    x: torch.Tensor, src: Union[int, Sequence[int]], dst: Union[int, Sequence[int]]
) -> torch.Tensor:
    # get original indices
    indices = list(range(x.ndim))
    len_indices = len(indices)
    for s, d in zip(ensure_tuple(src), ensure_tuple(dst)):
        # make src and dst positive
        # remove desired index and insert it in new position
        pos_s = len_indices + s if s < 0 else s
        pos_d = len_indices + d if d < 0 else d
        indices.pop(pos_s)
        indices.insert(pos_d, pos_s)
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
        result = torch.clamp(a, a_min, a_max)
    return result


def percentile(
    x: NdarrayOrTensor, q, dim: Optional[int] = None, keepdim: bool = False, **kwargs
) -> Union[NdarrayOrTensor, float, int]:
    """`np.percentile` with equivalent implementation for torch.

    Pytorch uses `quantile`, but this functionality is only available from v1.7.
    For earlier methods, we calculate it ourselves. This doesn't do interpolation,
    so is the equivalent of ``numpy.percentile(..., interpolation="nearest")``.
    For more details, please refer to:
    https://pytorch.org/docs/stable/generated/torch.quantile.html.
    https://numpy.org/doc/stable/reference/generated/numpy.percentile.html.

    Args:
        x: input data
        q: percentile to compute (should in range 0 <= q <= 100)
        dim: the dim along which the percentiles are computed. default is to compute the percentile
            along a flattened version of the array. only work for numpy array or Tensor with PyTorch >= 1.7.0.
        keepdim: whether the output data has dim retained or not.
        kwargs: if `x` is numpy array, additional args for `np.percentile`, more details:
            https://numpy.org/doc/stable/reference/generated/numpy.percentile.html.

    Returns:
        Resulting value (scalar)
    """
    if np.isscalar(q):
        if not 0 <= q <= 100:  # type: ignore
            raise ValueError
    elif any(q < 0) or any(q > 100):
        raise ValueError
    result: Union[NdarrayOrTensor, float, int]
    if isinstance(x, np.ndarray):
        result = np.percentile(x, q, axis=dim, keepdims=keepdim, **kwargs)
    else:
        q = torch.tensor(q, device=x.device)
        if hasattr(torch, "quantile"):  # `quantile` is new in torch 1.7.0
            result = torch.quantile(x, q / 100.0, dim=dim, keepdim=keepdim)
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


def nonzero(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.nonzero` with equivalent implementation for torch.

    Args:
        x: array/tensor

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
        idx: index to unravel
        shape: shape of array/tensor

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
        idx: a sequence of indices to unravel
        shape: shape of array/tensor

    Returns:
        Stacked indices unravelled for given shape
    """
    lib_stack = torch.stack if isinstance(idx[0], torch.Tensor) else np.stack
    return lib_stack([unravel_index(i, shape) for i in idx])  # type: ignore


def ravel(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.ravel` with equivalent implementation for torch.

    Args:
        x: array/tensor to ravel

    Returns:
        Return a contiguous flattened array/tensor.
    """
    if isinstance(x, torch.Tensor):
        if hasattr(torch, "ravel"):  # `ravel` is new in torch 1.8.0
            return x.ravel()
        return x.flatten().contiguous()
    return np.ravel(x)


def any_np_pt(x: NdarrayOrTensor, axis: Union[int, Sequence[int]]) -> NdarrayOrTensor:
    """`np.any` with equivalent implementation for torch.

    For pytorch, convert to boolean for compatibility with older versions.

    Args:
        x: input array/tensor
        axis: axis to perform `any` over

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

    `torch.maximum` only available from pt>1.6, else use `torch.stack` and `torch.max`.

    Args:
        a: first array/tensor
        b: second array/tensor

    Returns:
        Element-wise maximum between two arrays/tensors.
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        # is torch and has torch.maximum (pt>1.6)
        if hasattr(torch, "maximum"):  # `maximum` is new in torch 1.7.0
            return torch.maximum(a, b)
        return torch.stack((a, b)).max(dim=0)[0]
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
        return np.cumsum(a, axis)
    if axis is None:
        return torch.cumsum(a[:], 0, **kwargs)
    return torch.cumsum(a, dim=axis, **kwargs)


def isfinite(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """`np.isfinite` with equivalent implementation for torch."""
    if not isinstance(x, torch.Tensor):
        return np.isfinite(x)
    return torch.isfinite(x)


def searchsorted(a: NdarrayOrTensor, v: NdarrayOrTensor, right=False, sorter=None, **kwargs) -> NdarrayOrTensor:
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


def repeat(a: NdarrayOrTensor, repeats: int, axis: Optional[int] = None, **kwargs) -> NdarrayOrTensor:
    """
    `np.repeat` with equivalent implementation for torch (`repeat_interleave`).

    Args:
        a: input data to repeat.
        repeats: number of repetitions for each element, repeats is broadcasted to fit the shape of the given axis.
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
        x: array/tensor

    """
    if isinstance(x, np.ndarray):
        return np.isnan(x)
    return torch.isnan(x)


def ascontiguousarray(x: NdarrayOrTensor, **kwargs) -> NdarrayOrTensor:
    """`np.ascontiguousarray` with equivalent implementation for torch (`contiguous`).

    Args:
        x: array/tensor
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
