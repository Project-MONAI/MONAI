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

import re
from typing import Any, Optional, Sequence, Tuple, Type, Union

import warnings
import numpy as np
import torch

from monai.config.type_definitions import DtypeLike, NdarrayTensor
from monai.utils import optional_import

cp, has_cp = optional_import("cupy")
cp_ndarray, _ = optional_import("cupy", name="ndarray")

__all__ = [
    "get_numpy_dtype_from_string",
    "get_torch_dtype_from_string",
    "dtype_torch_to_numpy",
    "dtype_numpy_to_torch",
    "get_equivalent_dtype",
    "convert_data_type",
    "get_dtype",
    "convert_to_cupy",
    "convert_to_numpy",
    "convert_to_tensor",
    "convert_to_meta_tensor",
    "convert_to_dst_type",
]


def get_numpy_dtype_from_string(dtype: str) -> np.dtype:
    """Get a numpy dtype (e.g., `np.float32`) from its string (e.g., `"float32"`)."""
    return np.zeros([], dtype=dtype).dtype  # type: ignore


def get_torch_dtype_from_string(dtype: str) -> torch.dtype:
    """Get a torch dtype (e.g., `torch.float32`) from its string (e.g., `"float32"`)."""
    return dtype_numpy_to_torch(get_numpy_dtype_from_string(dtype))


def dtype_torch_to_numpy(dtype: torch.dtype) -> np.dtype:
    """Convert a torch dtype to its numpy equivalent."""
    return torch.zeros([], dtype=dtype).numpy().dtype  # type: ignore


def dtype_numpy_to_torch(dtype: np.dtype) -> torch.dtype:
    """Convert a numpy dtype to its torch equivalent."""
    return torch.from_numpy(np.zeros([], dtype=dtype)).dtype


def get_equivalent_dtype(dtype, data_type):
    """Convert to the `dtype` that corresponds to `data_type`.

    The input dtype can also be a string. e.g., `"float32"` becomes `torch.float32` or
    `np.float32` as necessary.

    Example::

        im = torch.tensor(1)
        dtype = get_equivalent_dtype(np.float32, type(im))

    """
    if dtype is None:
        return None
    if data_type is torch.Tensor or data_type.__name__ == "MetaTensor":
        if isinstance(dtype, torch.dtype):
            # already a torch dtype and target `data_type` is torch.Tensor
            return dtype
        return dtype_numpy_to_torch(dtype)
    if not isinstance(dtype, torch.dtype):
        # assuming the dtype is ok if it is not a torch dtype and target `data_type` is not torch.Tensor
        return dtype
    return dtype_torch_to_numpy(dtype)


def get_dtype(data: Any):
    """Get the dtype of an image, or if there is a sequence, recursively call the method on the 0th element.

    This therefore assumes that in a `Sequence`, all types are the same.
    """
    if hasattr(data, "dtype"):
        return data.dtype
    # need recursion
    if isinstance(data, Sequence):
        return get_dtype(data[0])
    # objects like float don't have dtype, so return their type
    return type(data)


def convert_to_tensor(
    data, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None, wrap_sequence: bool = False
):
    """
    Utility to convert the input data to a PyTorch Tensor. If passing a dictionary, list or tuple,
    recursively check every item and convert it to PyTorch Tensor.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a Tensor if applicable.
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[tensor(1), tensor(2)]`. If `True`, then `[1, 2]` -> `tensor([1, 2])`.

    """
    # avoids circular import
    from monai.data.meta_tensor import MetaTensor

    if isinstance(data, torch.Tensor):
        if isinstance(data, MetaTensor):
            if data.applied_operations:
                raise ValueError(
                    f"cannot convert a MetaTensor with applied operations to a Tensor. Got{data.applied_operations}"
                    "please reset the applied operations before converting it to a Tensor.")
            data = data.as_tensor()
        return data.to(dtype=dtype, device=device, memory_format=torch.contiguous_format)  # type: ignore
    if isinstance(data, np.ndarray):
        # skip array of string classes and object, refer to:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/_utils/collate.py#L13
        if re.search(r"[SaUO]", data.dtype.str) is None:
            # numpy array with 0 dims is also sequence iterable,
            # `ascontiguousarray` will add 1 dim if img has no dim, so we only apply on data with dims
            if data.ndim > 0:
                data = np.ascontiguousarray(data)
            return torch.as_tensor(data, dtype=dtype, device=device)  # type: ignore
    elif (has_cp and isinstance(data, cp_ndarray)) or isinstance(data, (float, int, bool)):
        return torch.as_tensor(data, dtype=dtype, device=device)  # type: ignore
    elif isinstance(data, list):
        list_ret = [convert_to_tensor(i, dtype=dtype, device=device) for i in data]
        return torch.as_tensor(list_ret, dtype=dtype, device=device) if wrap_sequence else list_ret  # type: ignore
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_tensor(i, dtype=dtype, device=device) for i in data)
        return torch.as_tensor(tuple_ret, dtype=dtype, device=device) if wrap_sequence else tuple_ret  # type: ignore
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v, dtype=dtype, device=device) for k, v in data.items()}

    return data


def convert_to_meta_tensor(
    data, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None, wrap_sequence: bool = False
):
    """
    Utility to convert the input data to a MetaTensor. If passing a dictionary, list or tuple,
    recursively check every item and convert it to MetaTensor.

    Args:
        data: input data can be MetaTensor, PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a Tensor if applicable.
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[tensor(1), tensor(2)]`. If `True`, then `[1, 2]` -> `tensor([1, 2])`.

    """
    # avoids circular import
    from monai.data.meta_tensor import MetaTensor

    if isinstance(data, torch.Tensor):
        out = data.to(dtype=dtype, device=device, memory_format=torch.contiguous_format)  # type: ignore
        if not isinstance(out, MetaTensor):
            out = MetaTensor(out)
        return out
    if isinstance(data, np.ndarray):
        # skip array of string classes and object, refer to:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/_utils/collate.py#L13
        if re.search(r"[SaUO]", data.dtype.str) is None:
            # numpy array with 0 dims is also sequence iterable,
            # `ascontiguousarray` will add 1 dim if img has no dim, so we only apply on data with dims
            if data.ndim > 0:
                data = np.ascontiguousarray(data)
            return MetaTensor(torch.as_tensor(data, dtype=dtype, device=device))  # type: ignore
    elif (has_cp and isinstance(data, cp_ndarray)) or isinstance(data, (float, int, bool)):
        return MetaTensor(torch.as_tensor(data, dtype=dtype, device=device))  # type: ignore
    elif isinstance(data, list):
        list_ret = [convert_to_meta_tensor(i, dtype=dtype, device=device) for i in data]
        return (
            MetaTensor(torch.as_tensor(list_ret, dtype=dtype, device=device))  # type: ignore
            if wrap_sequence
            else list_ret
        )
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_meta_tensor(i, dtype=dtype, device=device) for i in data)
        return (
            MetaTensor(torch.as_tensor(tuple_ret, dtype=dtype, device=device))  # type: ignore
            if wrap_sequence
            else tuple_ret
        )
    elif isinstance(data, dict):
        return {k: convert_to_meta_tensor(v, dtype=dtype, device=device) for k, v in data.items()}

    return data


def convert_to_numpy(data, dtype: DtypeLike = None, wrap_sequence: bool = False):
    """
    Utility to convert the input data to a numpy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to numpy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to numpy arrays, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to numpy array.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().to(dtype=get_equivalent_dtype(dtype, torch.Tensor), device="cpu").numpy()
    elif has_cp and isinstance(data, cp_ndarray):
        data = cp.asnumpy(data).astype(dtype, copy=False)
    elif isinstance(data, (np.ndarray, float, int, bool)):
        data = np.asarray(data, dtype=dtype)
    elif isinstance(data, list):
        list_ret = [convert_to_numpy(i, dtype=dtype) for i in data]
        return np.asarray(list_ret) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_numpy(i, dtype=dtype) for i in data)
        return np.asarray(tuple_ret) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v, dtype=dtype) for k, v in data.items()}

    if isinstance(data, np.ndarray) and data.ndim > 0:
        data = np.ascontiguousarray(data)

    return data


def convert_to_cupy(data, dtype: Optional[np.dtype] = None, wrap_sequence: bool = False):
    """
    Utility to convert the input data to a cupy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to cupy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, cupy array, list, dictionary, int, float, bool, str, etc.
            Tensor, numpy array, cupy array, float, int, bool are converted to cupy arrays,
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to Cupy array, tt must be an argument of `numpy.dtype`,
            for more details: https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
    """

    # direct calls
    if isinstance(data, (cp_ndarray, np.ndarray, torch.Tensor, float, int, bool)):
        data = cp.asarray(data, dtype)
    elif isinstance(data, list):
        list_ret = [convert_to_cupy(i, dtype) for i in data]
        return cp.asarray(list_ret) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_cupy(i, dtype) for i in data)
        return cp.asarray(tuple_ret) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_cupy(v, dtype) for k, v in data.items()}
    # make it contiguous
    if not isinstance(data, cp.ndarray):
        raise ValueError(f"The input data type [{type(data)}] cannot be converted into cupy arrays!")

    if data.ndim > 0:
        data = cp.ascontiguousarray(data)
    return data


def convert_data_type(
    data: Any,
    output_type: Optional[Type[NdarrayTensor]] = None,
    device: Optional[torch.device] = None,
    dtype: Union[DtypeLike, torch.dtype] = None,
    wrap_sequence: bool = False,
) -> Tuple[NdarrayTensor, type, Optional[torch.device]]:
    """
    Convert to `torch.Tensor`/`np.ndarray` from `torch.Tensor`/`np.ndarray`/`float`/`int` etc.

    Args:
        data: data to be converted
        output_type: `torch.Tensor` or `np.ndarray` (if `None`, unchanged)
        device: if output is `torch.Tensor`, select device (if `None`, unchanged)
        dtype: dtype of output data. Converted to correct library type (e.g.,
            `np.float32` is converted to `torch.float32` if output type is `torch.Tensor`).
            If left blank, it remains unchanged.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
    Returns:
        modified data, orig_type, orig_device

    Note:
        When both `output_type` and `dtype` are specified with different backend
        (e.g., `torch.Tensor` and `np.float32`), the `output_type` will be used as the primary type,
        for example::

            >>> convert_data_type(1, torch.Tensor, dtype=np.float32)
            (1.0, <class 'torch.Tensor'>, None)

    """
    # avoids circular import
    from monai.data.meta_tensor import MetaTensor

    orig_type: type
    if isinstance(data, MetaTensor):
        orig_type = MetaTensor
    elif isinstance(data, torch.Tensor):
        orig_type = torch.Tensor
    elif isinstance(data, np.ndarray):
        orig_type = np.ndarray
    elif has_cp and isinstance(data, cp.ndarray):
        orig_type = cp.ndarray
    else:
        orig_type = type(data)

    orig_device = data.device if isinstance(data, torch.Tensor) else None

    output_type = output_type or orig_type

    dtype_ = get_equivalent_dtype(dtype, output_type)

    data_: NdarrayTensor

    if issubclass(output_type, MetaTensor):
        data_ = convert_to_meta_tensor(data, dtype=dtype_, device=device, wrap_sequence=wrap_sequence)
        return data_, orig_type, orig_device
    if issubclass(output_type, torch.Tensor):
        data_ = convert_to_tensor(data, dtype=dtype_, device=device, wrap_sequence=wrap_sequence)
        return data_, orig_type, orig_device
    if issubclass(output_type, np.ndarray):
        data_ = convert_to_numpy(data, dtype=dtype_, wrap_sequence=wrap_sequence)
        return data_, orig_type, orig_device
    elif has_cp and issubclass(output_type, cp.ndarray):
        data_ = convert_to_cupy(data, dtype=dtype_, wrap_sequence=wrap_sequence)
        return data_, orig_type, orig_device
    raise ValueError(f"Unsupported output type: {output_type}")


def convert_to_dst_type(
    src: Any, dst: NdarrayTensor, dtype: Union[DtypeLike, torch.dtype, None] = None, wrap_sequence: bool = False
) -> Tuple[NdarrayTensor, type, Optional[torch.device]]:
    """
    Convert source data to the same data type and device as the destination data.
    If `dst` is an instance of `torch.Tensor` or its subclass, convert `src` to `torch.Tensor` with the same data type as `dst`,
    if `dst` is an instance of `numpy.ndarray` or its subclass, convert to `numpy.ndarray` with the same data type as `dst`,
    otherwise, convert to the type of `dst` directly.

    Args:
        src: source data to convert type.
        dst: destination data that convert to the same data type as it.
        dtype: an optional argument if the target `dtype` is different from the original `dst`'s data type.
        wrap_sequence: if `False`, then lists will recursively call this function. E.g., `[1, 2]` -> `[array(1), array(2)]`.
            If `True`, then `[1, 2]` -> `array([1, 2])`.

    See Also:
        :func:`convert_data_type`
    """
    # avoids circular import
    from monai.data.meta_tensor import MetaTensor

    device = dst.device if isinstance(dst, torch.Tensor) else None
    if dtype is None:
        dtype = dst.dtype

    output_type: Any
    if isinstance(dst, MetaTensor):
        output_type = MetaTensor
    elif isinstance(dst, torch.Tensor):
        output_type = torch.Tensor
    elif isinstance(dst, np.ndarray):
        output_type = np.ndarray
    else:
        output_type = type(dst)
    return convert_data_type(data=src, output_type=output_type, device=device, dtype=dtype, wrap_sequence=wrap_sequence)


def convert_to_list(data: Union[Sequence, torch.Tensor, np.ndarray]) -> list:
    """
    Convert to list from `torch.Tensor`/`np.ndarray`/`list`/`tuple` etc.
    Args:
        data: data to be converted
    Returns:
        a list

    """
    return data.tolist() if isinstance(data, (torch.Tensor, np.ndarray)) else list(data)
