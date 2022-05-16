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

from copy import deepcopy
from typing import Sequence, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.data import box_mode
from monai.data.box_mode import (
    BoxMode,
    CenterSizeMode,
    CornerCornerMode_TypeA,
    CornerCornerMode_TypeB,
    CornerCornerMode_TypeC,
    CornerSizeMode,
)
from monai.utils import look_up_option
from monai.utils.enums import BoundingBoxMode
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

# TO_REMOVE = 0 if the bottom-right corner pixel/voxel is not included in the box,
#      i.e., when xmin=1, xmax=2, we have w = 1
# TO_REMOVE = 1  if the bottom-right corner pixel/voxel is included in the box,
#       i.e., when xmin=1, xmax=2, we have w = 2
# Currently, only `TO_REMOVE = 0.` is supported
TO_REMOVE = box_mode.TO_REMOVE

# We support the conversion between several box modes, i.e., representation of a bounding box
# BOXMODE_MAPPING maps string box mode to the corresponding BoxMode class
BOXMODE_MAPPING = {
    BoundingBoxMode.XYXY: CornerCornerMode_TypeA(),  # [xmin, ymin, xmax, ymax]
    BoundingBoxMode.XYZXYZ: CornerCornerMode_TypeA(),  # [xmin, ymin, zmin, xmax, ymax, zmax]
    BoundingBoxMode.XXYY: CornerCornerMode_TypeB(),  # [xmin, xmax, ymin, ymax]
    BoundingBoxMode.XXYYZZ: CornerCornerMode_TypeB(),  # [xmin, xmax, ymin, ymax, zmin, zmax]
    BoundingBoxMode.XYXYZZ: CornerCornerMode_TypeC(),  # [xmin, ymin, xmax, ymax, zmin, zmax]
    BoundingBoxMode.XYWH: CornerSizeMode(),  # [xmin, ymin, xsize, ysize]
    BoundingBoxMode.XYZWHD: CornerSizeMode(),  # [xmin, ymin, zmin, xsize, ysize, zsize]
    BoundingBoxMode.CCWH: CenterSizeMode(),  # [xcenter, ycenter, xsize, ysize]
    BoundingBoxMode.CCCWHD: CenterSizeMode(),  # [xcenter, ycenter, zcenter, xsize, ysize, zsize]
}
# The standard box mode we use in all the box util functions
StandardMode = CornerCornerMode_TypeA


def get_boxmode(mode: Union[str, BoxMode, None] = None) -> BoxMode:
    """
    This function returns BoxMode object from giving mode according to BOXMODE_MAPPING
    Args:
        mode: source box mode. If mode is not given, this func will assume mode is StandardMode()
    Returns:
        BoxMode object

    Example:
        mode = "xyzxyz"
        get_boxmode(mode) will return CornerCornerMode_TypeA()
    """
    if isinstance(mode, BoxMode):
        return mode
    elif isinstance(mode, str):
        return BOXMODE_MAPPING[mode]
    elif mode is None:
        return StandardMode()
    else:
        raise ValueError("mode has to be chosen from [str, BoxMode, None].")


def convert_to_list(in_sequence: Union[Sequence, torch.Tensor, np.ndarray]) -> list:
    """
    Convert a torch.Tensor, or np array input to list
    Args:
        in_sequence: Sequence or torch.Tensor or np.ndarray
    Returns:
        a list

    """
    return in_sequence.tolist() if isinstance(in_sequence, (torch.Tensor, np.ndarray)) else list(in_sequence)


def get_dimension(
    boxes: Union[torch.Tensor, np.ndarray, None] = None,
    spatial_size: Union[Sequence[int], torch.Tensor, np.ndarray, None] = None,
) -> int:
    """
    Get spatial dimension for the giving setting.
    Missing input is allowed. But at least one of the input value should be given.
    It raises ValueError if the dimensions of multiple inputs do not match with each other.
    Args:
        boxes: bounding box, Nx4 or Nx6 torch tensor or ndarray
        spatial_size: Length of 2 or 3. Data format is list, or np.ndarray, or tensor of int
    Returns:
        spatial_dimension: 2 or 3

    Example:
        boxes = torch.ones(10,6)
        get_dimension(boxes, spatial_size=[100,200,200]) will return 3
        get_dimension(boxes) will return 3
    """
    spatial_dims_set = set()
    if spatial_size is not None:
        spatial_dims_set.add(len(spatial_size))
    if boxes is not None:
        spatial_dims_set.add(int(boxes.shape[1] / 2))
    spatial_dims_list = list(spatial_dims_set)
    if len(spatial_dims_list) == 0:
        raise ValueError("At least one of boxes, spatial_size, and mode needs to be non-empty.")
    elif len(spatial_dims_list) == 1:
        spatial_dims = int(spatial_dims_list[0])
        spatial_dims = look_up_option(spatial_dims, supported=[2, 3])
        return int(spatial_dims)
    else:
        raise ValueError("The dimension of boxes, spatial_size, mode should match with each other.")


def convert_box_mode(
    boxes: NdarrayOrTensor, src_mode: Union[str, BoxMode, None] = None, dst_mode: Union[str, BoxMode, None] = None
) -> NdarrayOrTensor:
    """
    This function converts the boxes in src_mode to the dst_mode
    Args:
        boxes: source bounding box, Nx4 or Nx6 torch tensor or ndarray
        src_mode: source box mode. If mode is not given, this func will assume mode is StandardMode()
        dst_mode: target box mode. If mode is not given, this func will assume mode is StandardMode()
    Returns:
        boxes_dst: bounding box with target mode, does not share memory with original boxes

    Example:
        boxes = torch.ones(10,6)
        box_convert_mode(boxes=boxes, src_mode="xyzxyz", dst_mode="cccwhd")
    """

    # if mode not changed, return original box
    src_boxmode = get_boxmode(src_mode)
    dst_boxmode = get_boxmode(dst_mode)
    if type(src_boxmode) is type(dst_boxmode):
        return deepcopy(boxes)
    # convert mode
    else:
        # convert numpy to tensor if needed
        boxes_t, *_ = convert_data_type(boxes, torch.Tensor)

        corners = src_boxmode.box_to_corner(boxes_t)
        boxes_t_dst = dst_boxmode.corner_to_box(corners)

        # convert tensor back to numpy if needed
        boxes_dst, *_ = convert_to_dst_type(src=boxes_t_dst, dst=boxes)
        return boxes_dst


def convert_box_to_standard_mode(boxes: NdarrayOrTensor, mode: Union[str, BoxMode, None] = None) -> NdarrayOrTensor:
    """
    Convert given boxes to standard mode
    Args:
        boxes: source bounding box, Nx4 or Nx6 torch tensor or ndarray
        mode: source box mode. If mode is not given, this func will assume mode is StandardMode()
    Returns:
        boxes_standard: bounding box with standard mode, does not share memory with original boxes

    Example:
        boxes = torch.ones(10,6)
        box_convert_standard_mode(boxes=boxes, mode="xxyyzz")
    """
    return convert_box_mode(boxes=boxes, src_mode=mode, dst_mode=StandardMode())
