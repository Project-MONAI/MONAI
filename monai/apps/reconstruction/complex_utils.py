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
This script contains utility functions for complex-value PyTorch tensor.
"""

import re
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.type_conversion import convert_to_numpy, convert_to_tensor


def convert_to_tensor_complex(
    data,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    wrap_sequence: bool = True,
    track_meta: bool = False,
) -> Tensor:
    """
    Convert complex-valued data to a 2-channel PyTorch tensor.
    The real and imaginary parts are stacked along the last dimension.
    This function relies on 'monai.utils.type_conversion.convert_to_tensor'

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, int, and float.
            will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
            for list, convert every item to a Tensor if applicable.
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[tensor(1), tensor(2)]`. If `True`, then `[1, 2]` -> `tensor([1, 2])`.
        track_meta: whether to track the meta information, if `True`, will convert to `MetaTensor`.
            default to `False`.

    Returns:
        PyTorch version of the data

    Example:
        .. code-block:: python

            import numpy as np
            data = np.array([ [1+1j, 1-1j], [2+2j, 2-2j] ])
            # the following line prints (2,2)
            print(data.shape)
            # the following line prints torch.Size([2, 2, 2])
            print(convert_to_tensor_complex(data).shape)
    """
    # if data is not complex, just turn it into a tensor
    if isinstance(data, Tensor):
        if not torch.is_complex(data):
            converted_data: Tensor = convert_to_tensor(
                data, dtype=dtype, device=device, wrap_sequence=wrap_sequence, track_meta=track_meta
            )
            return converted_data
    else:
        if not np.iscomplexobj(data):
            converted_data = convert_to_tensor(
                data, dtype=dtype, device=device, wrap_sequence=wrap_sequence, track_meta=track_meta
            )
            return converted_data

    # if data is complex, turn its stacked version into a tensor
    if isinstance(data, torch.Tensor):
        data = torch.stack([data.real, data.imag], dim=-1)

    elif isinstance(data, np.ndarray):
        if re.search(r"[SaUO]", data.dtype.str) is None:
            # numpy array with 0 dims is also sequence iterable,
            # `ascontiguousarray` will add 1 dim if img has no dim, so we only apply on data with dims
            if data.ndim > 0:
                data = np.ascontiguousarray(data)
            data = np.stack((data.real, data.imag), axis=-1)

    elif isinstance(data, (float, int)):
        data = [[data.real, data.imag]]

    elif isinstance(data, list):
        data = convert_to_numpy(data, wrap_sequence=True)
        data = np.stack((data.real, data.imag), axis=-1).tolist()

    converted_data = convert_to_tensor(
        data, dtype=dtype, device=device, wrap_sequence=wrap_sequence, track_meta=track_meta
    )
    return converted_data


def complex_abs(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    Compute the absolute value of a complex array.

    Args:
        x: Input array/tensor with 2 channels in the last dimension representing real and imaginary parts.

    Returns:
        Absolute value along the last dimention

    Example:
        .. code-block:: python

            import numpy as np
            x = np.array([3,4])[np.newaxis]
            # the following line prints 5
            print(complex_abs(x))
    """
    if x.shape[-1] != 2:
        raise ValueError(f"x.shape[-1] is not 2 ({x.shape[-1]}).")
    return (x[..., 0] ** 2 + x[..., 1] ** 2) ** 0.5


def complex_mul(x: NdarrayOrTensor, y: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    Compute complex-valued multiplication. Supports Ndim inputs with last dim equal to 2 (real/imaginary channels)

    Args:
        x: Input array/tensor with 2 channels in the last dimension representing real and imaginary parts.
        y: Input array/tensor with 2 channels in the last dimension representing real and imaginary parts.

    Returns:
        Complex multiplication of x and y

    Example:
        .. code-block:: python

            import numpy as np
            x = np.array([[1,2],[3,4]])
            y = np.array([[1,1],[1,1]])
            # the following line prints array([[-1,  3], [-1,  7]])
            print(complex_mul(x,y))
    """
    if x.shape[-1] != 2 or y.shape[-1] != 2:
        raise ValueError(f"last dim must be 2, but x.shape[-1] is {x.shape[-1]} and y.shape[-1] is {y.shape[-1]}.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    if isinstance(x, Tensor):
        return torch.stack((re, im), dim=-1)  # type: ignore
    else:
        mult: np.ndarray = np.stack((re, im), axis=-1)
        return mult


def complex_conj(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    Compute complex conjugate of a tensor. Supports Ndim inputs with last dim equal to 2 (real/imaginary channels)

    Args:
        x: Input array/tensor with 2 channels in the last dimension representing real and imaginary parts.

    Returns:
        Complex conjugate of x

    Example:
        .. code-block:: python

            import numpy as np
            x = np.array([[1,2],[3,4]])
            # the following line prints array([[ 1, -2], [ 3, -4]])
            print(complex_conj(x))
    """
    if x.shape[-1] != 2:
        raise ValueError(f"last dim must be 2, but x.shape[-1] is {x.shape[-1]}.")

    if isinstance(x, Tensor):
        return torch.stack((x[..., 0], -x[..., 1]), dim=-1)
    else:
        np_conj: np.ndarray = np.stack((x[..., 0], -x[..., 1]), axis=-1)
        return np_conj
