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

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
import copy
import numpy as np
import skimage
from scipy import stats

from monai.utils import (
    TransformBackends,
    convert_data_type,
    convert_to_dst_type,
    get_equivalent_dtype,
    ensure_tuple_rep,
)
from monai.config import DtypeLike, NdarrayOrTensor
from monai.bundle import ConfigParser
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

def erode(mask: NdarrayOrTensor, filter_size: int|Sequence[int] = 3, pad_value: float = 1.0) -> NdarrayOrTensor:
    """
    Erode 2D/3D binary mask.

    Args:
        mask: input 2D/3D binary mask, [N,C,M,N] or [N,C,M,N,P] torch tensor or ndarray.
        filter_size: erosion filter size, has to be odd numbers, default to be 3.
        pad_value: the filled value for padding. We need to pad the input before filtering to keep the output with the same size as input. Usually use default value and not changed.

    Return:
        eroded mask, [N,C,M,N] or [N,C,M,N,P] torch tensor or ndarray.
    """
    mask_t, *_ = convert_data_type(mask, torch.Tensor)
    res_mask_t = erode_t(mask_t, filter_size=filter_size, pad_value=pad_value)
    res_mask: NdarrayOrTensor
    res_mask, *_ = convert_to_dst_type(src=res_mask_t, dst=mask)
    return res_mask

def dilate(mask: NdarrayOrTensor, filter_size: int|Sequence[int] = 3, pad_value: float = 0.0) -> NdarrayOrTensor:
    """
    Dilate 2D/3D binary mask.

    Args:
        mask: input 2D/3D binary mask, [N,C,M,N] or [N,C,M,N,P] torch tensor or ndarray.
        filter_size: dilation filter size, has to be odd numbers, default to be 3.
        pad_value: the filled value for padding. We need to pad the input before filtering to keep the output with the same size as input. Usually use default value and not changed.

    Return:
        dilated mask, [N,C,M,N] or [N,C,M,N,P] torch tensor or ndarray.
    """
    mask_t, *_ = convert_data_type(mask, torch.Tensor)
    res_mask_t = dilate_t(mask_t, filter_size=filter_size, pad_value=pad_value)
    res_mask: NdarrayOrTensor
    res_mask, *_ = convert_to_dst_type(src=res_mask_t, dst=mask)
    return res_mask

def get_morphological_filter_result_t(mask_t: Tensor, filter_size: int|Sequence[int], pad_value: float) -> Tensor:
    """
    Get morphological filter result for 2D/3D mask with data type as torch tensor.

    Args:
        mask_t: input 2D/3D binary mask, [N,C,M,N] or [N,C,M,N,P] torch tensor.
        filter_size: morphological filter size, has to be odd numbers.
        pad_value: the filled value for padding. We need to pad the input before filtering to keep the output with the same size as input.

    Return:
        Morphological filter result mask, [N,C,M,N] or [N,C,M,N,P] torch tensor
    """
    spatial_dims = len(mask_t.shape)-2
    if spatial_dims not in [2,3]:
        raise ValueError(f"spatial_dims must be either 2 or 3, yet got spatial_dims={spatial_dims} for mask tensor with shape of {mask_t.shape}.")
    
    # Define the structuring element
    filter_size = ensure_tuple_rep(filter_size, spatial_dims)
    if any(size % 2 == 0 for size in filter_size):
        raise ValueError(f"All dimensions in filter_size must be odd numbers, yet got {filter_size}.")
    
    filter_shape = [mask_t.shape[1],mask_t.shape[1]]+list(filter_size)
    structuring_element = torch.ones(filter_shape).to(
        mask_t.device
    )

    # Pad the input tensor to handle border pixels
    # Calculate padding size
    pad_size = []
    for size in filter_size:
        pad_size.extend([size // 2, size // 2])
    
    input_padded = F.pad(
        mask_t.float(),
        pad_size,
        mode="constant",
        value=pad_value,
    )

    # Apply filter operation
    if spatial_dims == 2:
        output = F.conv2d(input_padded, structuring_element, padding=0)/torch.sum(structuring_element[0,...])
    if spatial_dims == 3:
        output = F.conv3d(input_padded, structuring_element, padding=0)/torch.sum(structuring_element[0,...])

    return output
    
def erode_t(mask_t: Tensor, filter_size: int|Sequence[int] = 3, pad_value: float = 1.0) -> Tensor:
    """
    Erode 2D/3D binary mask with data type as torch tensor.

    Args:
        mask_t: input 2D/3D binary mask, [N,C,M,N] or [N,C,M,N,P] torch tensor.
        filter_size: erosion filter size, has to be odd numbers, default to be 3.
        pad_value: the filled value for padding. We need to pad the input before filtering to keep the output with the same size as input. Usually use default value and not changed.

    Return:
        eroded mask, [N,C,M,N] or [N,C,M,N,P] torch tensor
    """
    
    output = get_morphological_filter_result_t(mask_t, filter_size, pad_value)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output == 1.0, 1.0, 0.0)

    return output


def dilate_t(mask_t: Tensor, filter_size: int|Sequence[int] = 3, pad_value: float = 0.0) -> Tensor:
    """
    Dilate 2D/3D binary mask with data type as torch tensor.

    Args:
        mask_t: input 2D/3D binary mask, [N,C,M,N] or [N,C,M,N,P] torch tensor.
        filter_size: dilation filter size, has to be odd numbers, default to be 3.
        pad_value: the filled value for padding. We need to pad the input before filtering to keep the output with the same size as input. Usually use default value and not changed.

    Return:
        dilated mask, [N,C,M,N] or [N,C,M,N,P] torch tensor
    """
    output = get_morphological_filter_result_t(mask_t, filter_size, pad_value)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output > 0, 1.0, 0.0)

    return output


