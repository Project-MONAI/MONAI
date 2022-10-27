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

import itertools as it
from typing import Optional, Sequence, Union

import numpy as np
import torch

from monai.config import DtypeLike
from monai.data.meta_tensor import MetaTensor
from monai.transforms.meta_matrix import Matrix, MatrixFactory, MetaMatrix, matmul
from monai.transforms.spatial.array import Affine
from monai.transforms.utils import dtypes_to_str_or_identity, get_backend_from_tensor_like, get_device_from_tensor_like
from monai.utils import GridSampleMode, GridSamplePadMode

__all__ = ["apply"]

# TODO: This should move to a common place to be shared with dictionary
# from monai.utils.type_conversion import dtypes_to_str_or_identity

GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]
DtypeSequence = Union[Sequence[DtypeLike], DtypeLike]


# TODO: move to mapping_stack.py
def extents_from_shape(shape, dtype=np.float64):
    extents = [[0, shape[i]] for i in range(1, len(shape))]

    extents = it.product(*extents)
    return [np.asarray(e + (1,), dtype=dtype) for e in extents]


# TODO: move to mapping_stack.py
def shape_from_extents(
    src_shape: Sequence, extents: Union[Sequence[np.ndarray], Sequence[torch.Tensor], np.ndarray, torch.Tensor]
):
    if isinstance(extents, (list, tuple)):
        if isinstance(extents[0], np.ndarray):
            aextents = np.asarray(extents)
            aextents = torch.from_numpy(aextents)
        else:
            aextents = torch.stack(extents)
    else:
        if isinstance(extents, np.ndarray):
            aextents = torch.from_numpy(extents)
        else:
            aextents = extents

    mins = aextents.min(axis=0)[0]
    maxes = aextents.max(axis=0)[0]
    values = torch.round(maxes - mins).type(torch.IntTensor)[:-1]
    return torch.cat((torch.as_tensor([src_shape[0]]), values))


def metadata_is_compatible(value_1, value_2):
    if value_1 is None:
        return True
    else:
        if value_2 is None:
            return True
        return value_1 == value_2


def metadata_dtype_is_compatible(value_1, value_2):
    if value_1 is None:
        return True
    else:
        if value_2 is None:
            return True

    # if we are here, value_1 and value_2 are both set
    # TODO: this is not a good enough solution
    value_1_ = dtypes_to_str_or_identity(value_1)
    value_2_ = dtypes_to_str_or_identity(value_2)
    return value_1_ == value_2_


def starting_matrix_and_extents(matrix_factory, data):
    # set up the identity matrix and metadata
    cumulative_matrix = matrix_factory.identity()
    cumulative_extents = extents_from_shape(data.shape)
    return cumulative_matrix, cumulative_extents


def prepare_args_dict_for_apply(cur_mode, cur_padding_mode, cur_device, cur_dtype):
    kwargs = {}
    if cur_mode is not None:
        kwargs["mode"] = cur_mode
    if cur_padding_mode is not None:
        kwargs["padding_mode"] = cur_padding_mode
    if cur_device is not None:
        kwargs["device"] = cur_device
    if cur_dtype is not None:
        kwargs["dtype"] = cur_dtype

    return kwargs


def matrix_from_matrix_container(matrix):
    if isinstance(matrix, MetaMatrix):
        return matrix.matrix.data
    elif isinstance(matrix, Matrix):
        return matrix.data
    else:
        return matrix


def apply(data: Union[torch.Tensor, MetaTensor, dict], pending: Optional[Union[dict, list]] = None):
    """
    This method applies pending transforms to tensors.

    Args:
        data: A torch Tensor, monai MetaTensor, or a dictionary containing Tensors / MetaTensors
        pending: Optional arg containing pending transforms. This must be set if data is a Tensor
        or dictionary of Tensors, but is optional if data is a MetaTensor / dictionary of
        MetaTensors.
    """
    if isinstance(data, dict):
        rd = dict()
        for k, v in data.items():
            result = apply(v, pending)
            rd[k] = result
        return rd

    if isinstance(data, MetaTensor) or pending is not None:
        pending_ = [] if pending is None else pending
    else:
        pending_ = data.pending_transforms

    if len(pending_) == 0:
        return data

    dim_count = len(data.shape) - 1
    matrix_factory = MatrixFactory(dim_count, get_backend_from_tensor_like(data), get_device_from_tensor_like(data))

    # set up the identity matrix and metadata
    cumulative_matrix, cumulative_extents = starting_matrix_and_extents(matrix_factory, data)

    # set the various resampling parameters to an initial state
    cur_mode = None
    cur_padding_mode = None
    cur_device = None
    cur_dtype = None
    cur_shape = data.shape

    for meta_matrix in pending_:
        next_matrix = meta_matrix.data
        # print("intermediate matrix\n", matrix_from_matrix_container(cumulative_matrix))
        cumulative_matrix = matmul(cumulative_matrix, next_matrix)
        cumulative_extents = [matmul(e, cumulative_matrix) for e in cumulative_extents]

        new_mode = meta_matrix.metadata.get("mode", None)
        new_padding_mode = meta_matrix.metadata.get("padding_mode", None)
        new_device = meta_matrix.metadata.get("device", None)
        new_dtype = meta_matrix.metadata.get("dtype", None)
        new_shape = meta_matrix.metadata.get("shape_override", None)

        mode_compat = metadata_is_compatible(cur_mode, new_mode)
        padding_mode_compat = metadata_is_compatible(cur_padding_mode, new_padding_mode)
        device_compat = metadata_is_compatible(cur_device, new_device)
        dtype_compat = metadata_dtype_is_compatible(cur_dtype, new_dtype)

        if mode_compat is False or padding_mode_compat is False or device_compat is False or dtype_compat is False:
            # carry out an intermediate resample here due to incompatibility between arguments
            kwargs = prepare_args_dict_for_apply(cur_mode, cur_padding_mode, cur_device, cur_dtype)

            cumulative_matrix_ = matrix_from_matrix_container(cumulative_matrix)
            a = Affine(norm_coords=False, affine=cumulative_matrix_, **kwargs)
            data, _ = a(img=data)

            cumulative_matrix, cumulative_extents = starting_matrix_and_extents(matrix_factory, data)

        cur_mode = cur_mode if new_mode is None else new_mode
        cur_padding_mode = cur_padding_mode if new_padding_mode is None else new_padding_mode
        cur_device = cur_device if new_device is None else new_device
        cur_dtype = cur_dtype if new_dtype is None else new_dtype
        cur_shape = cur_shape if new_shape is None else new_shape

    kwargs = prepare_args_dict_for_apply(cur_mode, cur_padding_mode, cur_device, cur_dtype)

    cumulative_matrix_ = matrix_from_matrix_container(cumulative_matrix)

    # print(f"applying with cumulative matrix\n {cumulative_matrix_}")
    a = Affine(norm_coords=False, affine=cumulative_matrix_, spatial_size=cur_shape[1:], normalized=False, **kwargs)
    data, tx = a(img=data)
    data.clear_pending_transforms()

    return data
