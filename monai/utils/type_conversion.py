from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config.type_definitions import DtypeLike
from monai.utils.enums import DataObjects

__all__ = [
    "dtype_torch_to_numpy",
    "dtype_numpy_to_torch",
    "get_equivalent_dtype",
    "convert_data_type",
    "get_dtype",
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
    return _torch_to_np_dtype[dtype]


def dtype_numpy_to_torch(dtype):
    """Convert a numpy dtype to its torch equivalent."""
    # np dtypes can be given as np.float32 and np.dtype(np.float32) so unify them
    dtype = np.dtype(dtype) if type(dtype) is type else dtype
    return _np_to_torch_dtype[dtype]


def get_equivalent_dtype(dtype, data_type):
    """Convert to the `dtype` that corresponds to `data_type`.
    Example:
        im = torch.tensor(1)
        dtype = dtype_convert(np.float32, type(im))
    """
    if data_type is torch.Tensor:
        if type(dtype) is torch.dtype:
            return dtype
        return dtype_numpy_to_torch(dtype)
    else:
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


def convert_data_type(
    data: Any,
    output_type: Optional[type] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
) -> Tuple[DataObjects.Images, type, Optional[torch.device]]:
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
        if orig_type is np.ndarray:
            if (np.array(data.strides) < 0).any():  # copy if -ve stride
                data = data.copy()
            data = torch.as_tensor(data if data.ndim == 0 else np.ascontiguousarray(data))
        else:
            data = torch.as_tensor(data)
        if dtype != data.dtype:
            data = data.to(dtype)  # type: ignore
    elif output_type is np.ndarray:
        if orig_type is torch.Tensor:
            data = data.detach().cpu().numpy()  # type: ignore
        else:
            data = np.array(data)
        if dtype != data.dtype:
            data = data.astype(dtype)  # type: ignore

    if isinstance(data, torch.Tensor) and device is not None:
        data = data.to(device)

    return data, orig_type, orig_device
