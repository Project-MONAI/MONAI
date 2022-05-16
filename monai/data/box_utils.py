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

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Sequence, Tuple, Union

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
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_numpy, convert_to_tensor

# TO_REMOVE = 0 if in 'xxyy','xxyyzz' mode, the bottom-right corner is not included in the box,
#      i.e., when xmin=1, xmax=2, we have w = 1
# TO_REMOVE = 1  if in 'xxyy','xxyyzz' mode, the bottom-right corner is included in the box,
#       i.e., when xmin=1, xmax=2, we have w = 2
# Currently, only `TO_REMOVE = 0` is supported
TO_REMOVE = box_mode.TO_REMOVE

# We support the conversion between several box modes, i.e., representation of a bounding box
# BOXMODE_MAPPING maps string box mode to teh corresponding BoxMode class
BOXMODE_MAPPING = {
    "xyxy": CornerCornerMode_TypeA(),  # [xmin, ymin, xmax, ymax]
    "xyzxyz": CornerCornerMode_TypeA(),  # [xmin, ymin, zmin, xmax, ymax, zmax]
    "xxyy": CornerCornerMode_TypeB(),  # [xmin, xmax, ymin, ymax]
    "xxyyzz": CornerCornerMode_TypeB(),  # [xmin, xmax, ymin, ymax, zmin, zmax]
    "xyxyzz": CornerCornerMode_TypeC(),  # [xmin, ymin, xmax, ymax, zmin, zmax]
    "xywh": CornerSizeMode(),  # [xmin, ymin, xsize, ysize]
    "xyzwhd": CornerSizeMode(),  # [xmin, ymin, zmin, xsize, ysize, zsize]
    "ccwh": CenterSizeMode(),  # [xcenter, ycenter, xsize, ysize]
    "cccwhd": CenterSizeMode(),  # [xcenter, ycenter, zcenter, xsize, ysize, zsize]
}
# The standard box mode we use in all the box util functions
StandardMode = CornerCornerMode_TypeA


def get_boxmode(mode: Union[str, BoxMode, None] = None) -> BoxMode:
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


# def get_dimension(
#     boxes: Union[torch.Tensor, np.ndarray, None] = None,
#     image_size: Union[Sequence[int], torch.Tensor, np.ndarray, None] = None,
#     mode: Union[str, None] = None,
# ) -> int:
#     """
#     Get spatial dimension for the giving setting.
#     Missing input is allowed. But at least one of the input value should be given.
#     It raises ValueError if the dimensions of multiple inputs do not match with each other.
#     Args:
#         boxes: bounding box, Nx4 or Nx6 torch tensor or ndarray
#         image_size: Length of 2 or 3. Data format is list, or np.ndarray, or tensor of int
#         mode: box mode, choose from SUPPORTED_MODE. If mode is not given, this func will assume mode is STANDARD_MODE
#     Returns:
#         spatial_dimension: 2 or 3

#     Example:
#         boxes = torch.ones(10,6)
#         get_dimension(boxes, mode="xyxy") will raise ValueError
#         get_dimension(boxes, mode="xyzxyz") will return 3
#         get_dimension(boxes, mode="xyzxyz", image_size=[100,200,200]) will return 3
#         get_dimension(mode="xyzxyz") will return 3
#     """
#     spatial_dims_set = set()
#     if image_size is not None:
#         spatial_dims_set.add(len(image_size))
#     if mode is not None:
#         spatial_dims_set.add(int(len(mode) / 2))
#     if boxes is not None:
#         spatial_dims_set.add(int(boxes.shape[1] / 2))
#     spatial_dims_list = list(spatial_dims_set)
#     if len(spatial_dims_list) == 0:
#         raise ValueError("At least one of boxes, image_size, and mode needs to be non-empty.")
#     elif len(spatial_dims_list) == 1:
#         spatial_dims = int(spatial_dims_list[0])
#         spatial_dims = look_up_option(spatial_dims, supported=[2, 3])
#         return int(spatial_dims)
#     else:
#         raise ValueError("The dimension of boxes, image_size, mode should match with each other.")


# def check_box_mode(boxes: NdarrayOrTensor, mode: Union[str, None] = None):
#     """
#     This function checks whether the boxes is valid.
#     It ensures the box size is non-negative.
#     Args:
#         boxes: bounding box, Nx4 or Nx6 torch tensor or ndarray
#         mode: box mode, choose from SUPPORTED_MODE. If mode is not given, this func will assume mode is STANDARD_MODE
#     Returns:
#         raise Error is mode is not supported
#         return False if box has negative size
#         return True if no issue found

#     Example:
#         boxes = torch.ones(10,6)
#         check_box_mode(boxes, mode="cccwhd")
#     """
#     if mode is None:
#         mode = get_standard_mode(int(boxes.shape[1] / 2))
#     mode = look_up_option(mode, supported=SUPPORTED_MODE)
#     spatial_dims = get_dimension(boxes=boxes, mode=mode)

#     # we need box size to be non-negative
#     if mode in ["ccwh", "cccwhd", "xywh", "xyzwhd"]:
#         box_error = boxes[:, spatial_dims] < 0
#         for axis in range(1, spatial_dims):
#             box_error = box_error | (boxes[:, spatial_dims + axis] < 0)
#     elif mode in ["xxyy", "xxyyzz"]:
#         box_error = boxes[:, 1] < boxes[:, 0]
#         for axis in range(1, spatial_dims):
#             box_error = box_error | (boxes[:, 2 * axis + 1] < boxes[:, 2 * axis])
#     elif mode in ["xyxy", "xyzxyz"]:
#         box_error = boxes[:, spatial_dims] < boxes[:, 0]
#         for axis in range(1, spatial_dims):
#             box_error = box_error | (boxes[:, spatial_dims + axis] < boxes[:, axis])
#     else:
#         raise ValueError(f"Box mode {mode} not in {SUPPORTED_MODE}.")

#     if box_error.sum() > 0:
#         return False

#     return True


def box_convert_mode(
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

    # if not check_box_mode(boxes, src_mode):
    #     raise ValueError("Given boxes has invalid values. The box size must be non-negative.")

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


def box_convert_standard_mode(boxes: NdarrayOrTensor, mode: Union[str, BoxMode, None] = None) -> NdarrayOrTensor:
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
    return box_convert_mode(boxes=boxes, src_mode=mode, dst_mode=StandardMode())
