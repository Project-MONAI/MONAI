import re
from typing import Any, Optional, Sequence, Tuple, Type, Union, cast

import numpy as np
import torch

from monai.config.type_definitions import DtypeLike, NdarrayTensor
from monai.utils import optional_import

cp, has_cp = optional_import("cupy")
cp_ndarray, _ = optional_import("cupy", name="ndarray")

__all__ = [
    "dtype_torch_to_numpy",
    "dtype_numpy_to_torch",
    "get_equivalent_dtype",
    "convert_data_type",
    "get_dtype",
    "convert_to_numpy",
    "convert_to_tensor",
    "convert_to_dst_type",
]


_torch_to_np_dtype = {
    torch.bool: np.dtype(bool),
    torch.uint8: np.dtype(np.uint8),
    torch.int8: np.dtype(np.int8),
    torch.int16: np.dtype(np.int16),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.float16: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
    torch.complex64: np.dtype(np.complex64),
    torch.complex128: np.dtype(np.complex128),
}
_np_to_torch_dtype = {value: key for key, value in _torch_to_np_dtype.items()}


def dtype_torch_to_numpy(dtype):
    """Convert a torch dtype to its numpy equivalent."""
    if dtype not in _torch_to_np_dtype:
        raise ValueError(f"Unsupported torch to numpy dtype '{dtype}'.")
    return _torch_to_np_dtype[dtype]


def dtype_numpy_to_torch(dtype):
    """Convert a numpy dtype to its torch equivalent."""
    # np dtypes can be given as np.float32 and np.dtype(np.float32) so unify them
    dtype = np.dtype(dtype) if type(dtype) is type else dtype
    if dtype not in _np_to_torch_dtype:
        raise ValueError(f"Unsupported numpy to torch dtype '{dtype}'.")
    return _np_to_torch_dtype[dtype]


def get_equivalent_dtype(dtype, data_type):
    """Convert to the `dtype` that corresponds to `data_type`.
    Example:
        im = torch.tensor(1)
        dtype = get_equivalent_dtype(np.float32, type(im))
    """
    if data_type is torch.Tensor:
        if type(dtype) is torch.dtype:
            return dtype
        return dtype_numpy_to_torch(dtype)
    if type(dtype) is not torch.dtype:
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


def convert_to_tensor(data):
    """
    Utility to convert the input data to a PyTorch Tensor. If passing a dictionary, list or tuple,
    recursively check every item and convert it to PyTorch Tensor.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to Tensors, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a Tensor if applicable.

    """
    if isinstance(data, torch.Tensor):
        return data.contiguous()
    if isinstance(data, np.ndarray):
        # skip array of string classes and object, refer to:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/_utils/collate.py#L13
        if re.search(r"[SaUO]", data.dtype.str) is None:
            # numpy array with 0 dims is also sequence iterable,
            # `ascontiguousarray` will add 1 dim if img has no dim, so we only apply on data with dims
            return torch.as_tensor(data if data.ndim == 0 else np.ascontiguousarray(data))
    elif isinstance(data, (float, int, bool)):
        return torch.as_tensor(data)
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_tensor(i) for i in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_tensor(i) for i in data)

    return data


def convert_to_numpy(data):
    """
    Utility to convert the input data to a numpy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to numpy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to numpy arrays, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a numpy array if applicable.

    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif has_cp and isinstance(data, cp_ndarray):
        data = cp.asnumpy(data)
    elif isinstance(data, (float, int, bool)):
        data = np.asarray(data)
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_numpy(i) for i in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_numpy(i) for i in data)

    if isinstance(data, np.ndarray) and data.ndim > 0:
        data = np.ascontiguousarray(data)

    return data


def convert_data_type(
    data: Any,
    output_type: Optional[Type[NdarrayTensor]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
) -> Tuple[NdarrayTensor, type, Optional[torch.device]]:
    """
    Convert to `torch.Tensor`/`np.ndarray` from `torch.Tensor`/`np.ndarray`/`float`/`int` etc.

    Args:
        data: data to be converted
        output_type: `torch.Tensor` or `np.ndarray` (if blank, unchanged)
        device: if output is `torch.Tensor`, select device (if blank, unchanged)
        dtype: dtype of output data. Converted to correct library type (e.g.,
            `np.float32` is converted to `torch.float32` if output type is `torch.Tensor`).
            If left blank, it remains unchanged.
    Returns:
        modified data, orig_type, orig_device
    """
    orig_type = type(data)
    orig_device = data.device if isinstance(data, torch.Tensor) else None

    output_type = output_type or orig_type

    dtype = get_equivalent_dtype(dtype or get_dtype(data), output_type)

    if output_type is torch.Tensor:
        if orig_type is not torch.Tensor:
            data = convert_to_tensor(data)
        if dtype != data.dtype:
            data = data.to(dtype)  # type: ignore
        if device is not None:
            data = data.to(device)
        return cast(NdarrayTensor, data), orig_type, orig_device  # pytype: disable=invalid-annotation
    if output_type is np.ndarray:
        if orig_type is not np.ndarray:
            data = convert_to_numpy(data)
        if data is not None and dtype != data.dtype:
            data = data.astype(dtype)  # type: ignore
        return cast(NdarrayTensor, data), orig_type, orig_device  # pytype: disable=invalid-annotation
    raise ValueError(f"Unsupported output type: {output_type}")


def convert_to_dst_type(src: Any, dst: NdarrayTensor) -> Tuple[NdarrayTensor, type, Optional[torch.device]]:
    """
    Convert `src` to the same `torch.Tensor`/`np.ndarray` and data type as `dst`.

    See Also:
        :func:`convert_data_type`
    """
    device = None
    if isinstance(dst, torch.Tensor):
        device = dst.device
    return convert_data_type(data=src, output_type=type(dst), device=device, dtype=dst.dtype)
