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

import inspect
from copy import deepcopy
from typing import Sequence, Type, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.data import box_mode
from monai.data.box_mode import (
    BoxMode,
    CenterSizeMode,
    CornerCornerModeTypeA,
    CornerCornerModeTypeB,
    CornerCornerModeTypeC,
    CornerSizeMode,
)
from monai.utils import look_up_option
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

# TO_REMOVE = 0.0 if the bottom-right corner pixel/voxel is not included in the box,
#      i.e., when xmin=1., xmax=2., we have w = 1.
# TO_REMOVE = 1.0  if the bottom-right corner pixel/voxel is included in the box,
#       i.e., when xmin=1., xmax=2., we have w = 2.
# Currently, only `TO_REMOVE = 0.0` is supported
TO_REMOVE = box_mode.TO_REMOVE

# We support 2_d or 3-D bounding boxes
SUPPORTED_SPATIAL_DIMS = [2, 3]

# We support the conversion between several box modes, i.e., representation of a bounding box
SUPPORTED_MODES = [CornerCornerModeTypeA, CornerCornerModeTypeB, CornerCornerModeTypeC, CornerSizeMode, CenterSizeMode]
# The standard box mode we use in all the box util functions
StandardMode = CornerCornerModeTypeA


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
    corners: Union[Sequence, None] = None,
    spatial_size: Union[Sequence[int], torch.Tensor, np.ndarray, None] = None,
) -> int:
    """
    Get spatial dimension for the giving setting.
    Missing input is allowed. But at least one of the input value should be given.
    It raises ValueError if the dimensions of multiple inputs do not match with each other.
    Args:
        boxes: bounding box, Nx4 or Nx6 torch tensor or ndarray
        corners: corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch tensor or ndarray
        spatial_size: The spatial size of the image where the boxes are attached.
                len(spatial_size) should be in [2, 3].
    Returns:
        spatial_dims: 2 or 3

    Example:
        boxes = torch.ones(10,6)
        get_dimension(boxes, spatial_size=[100,200,200]) will return 3
        get_dimension(boxes) will return 3
    """
    spatial_dims_set = set()
    if spatial_size is not None:
        spatial_dims_set.add(len(spatial_size))
    if corners is not None:
        if len(corners) not in [4, 6]:
            raise ValueError(
                f"Currently we support only boxes with shape [N,4] or [N,6], got box corner tuple with length {len(corners)}."
            )
        spatial_dims_set.add(len(corners) // 2)
    if boxes is not None:
        if int(boxes.shape[1]) not in [4, 6]:
            raise ValueError(
                f"Currently we support only boxes with shape [N,4] or [N,6], got boxes with shape {boxes.shape}."
            )
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


def get_boxmode(mode: Union[str, BoxMode, Type[BoxMode], None] = None, *args, **kwargs) -> BoxMode:
    """
    This function returns BoxMode object from giving mode according to BOXMODE_MAPPING
    Args:
        mode: source box mode. If mode is not given, this func will assume mode is StandardMode
    Returns:
        BoxMode object

    Example:
        mode = "xyzxyz"
        get_boxmode(mode) will return CornerCornerModeTypeA()
    """
    if isinstance(mode, BoxMode):
        return mode
    elif inspect.isclass(mode) and issubclass(mode, BoxMode):
        return mode(*args, **kwargs)
    elif isinstance(mode, str):
        for m in SUPPORTED_MODES:
            for n in SUPPORTED_SPATIAL_DIMS:
                if m.get_name(n) == mode:
                    return m(*args, **kwargs)
    elif mode is None:
        return StandardMode(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported box mode: {mode}.")


def check_corners(corners: Sequence) -> bool:
    """
    check the validity for the given box corners
    Args:
        corners: corners of a box, 4-element or 6-element tuple, each element is a Nx1 torch tensor
        (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax)
    Returns:
        bool, whether the box is valid
    Example:
        corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
        check_corner(corners) will return True
    """
    spatial_dims = get_dimension(corners=corners)
    box_error = corners[spatial_dims] < corners[0]
    for axis in range(1, spatial_dims):
        box_error = box_error | (corners[spatial_dims + axis] < corners[axis])
    if box_error.sum() > 0:
        return False
    else:
        return True


def convert_box_mode(
    boxes: NdarrayOrTensor,
    src_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
    dst_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
) -> NdarrayOrTensor:
    """
    This function converts the boxes in src_mode to the dst_mode
    Args:
        boxes: source bounding box, Nx4 or Nx6 torch tensor or ndarray
        src_mode: source box mode. If mode is not given, this func will assume mode is StandardMode.
            It can be:
            #. str: choose from monai.utils.enums.BoundingBoxMode, including
                "xyxy": boxes has format [xmin, ymin, xmax, ymax]
                "xyzxyz": boxes has format [xmin, ymin, zmin, xmax, ymax, zmax]
                "xxyy": boxes has format [xmin, xmax, ymin, ymax]
                "xxyyzz": boxes has format [xmin, xmax, ymin, ymax, zmin, zmax]
                "xyxyzz": boxes has format [xmin, ymin, xmax, ymax, zmin, zmax]
                "xywh": boxes has format [xmin, ymin, xsize, ysize]
                "xyzwhd": boxes has format [xmin, ymin, zmin, xsize, ysize, zsize]
                "ccwh": boxes has format [xcenter, ycenter, xsize, ysize]
                "cccwhd": boxes has format [xcenter, ycenter, zcenter, xsize, ysize, zsize]
            #. BoxMode class: choose from
                CornerCornerModeTypeA: equivalent to "xyxy" or "xyzxyz"
                CornerCornerModeTypeB: equivalent to "xxyy" or "xxyyzz"
                CornerCornerModeTypeC: equivalent to "xyxy" or "xyxyzz"
                CornerSizeMode: equivalent to "xywh" or "xyzwhd"
                CenterSizeMode: equivalent to "ccwh" or "cccwhd"
            #. BoxMode instance
            #. None: will assume mode is StandardMode
        dst_mode: target box mode. If mode is not given, this func will assume mode is StandardMode.
            Data type same as src_mode.
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

        # convert boxes to corners
        corners = src_boxmode.boxes_to_corners(boxes_t)

        # check validity of corners
        if not check_corners(corners):
            raise ValueError("Given boxes has invalid values. The box size must be non-negative.")

        # convert corners to boxes
        boxes_t_dst = dst_boxmode.corners_to_boxes(corners)

        # convert tensor back to numpy if needed
        boxes_dst, *_ = convert_to_dst_type(src=boxes_t_dst, dst=boxes)
        return boxes_dst


def convert_box_to_standard_mode(
    boxes: NdarrayOrTensor, mode: Union[str, BoxMode, Type[BoxMode], None] = None
) -> NdarrayOrTensor:
    """
    Convert given boxes to standard mode
    Args:
        boxes: source bounding box, Nx4 or Nx6 torch tensor or ndarray
        mode: source box mode. If mode is not given, this func will assume mode is StandardMode
        It can be:
            #. str: choose from monai.utils.enums.BoundingBoxMode, including
                "xyxy": boxes has format [xmin, ymin, xmax, ymax]
                "xyzxyz": boxes has format [xmin, ymin, zmin, xmax, ymax, zmax]
                "xxyy": boxes has format [xmin, xmax, ymin, ymax]
                "xxyyzz": boxes has format [xmin, xmax, ymin, ymax, zmin, zmax]
                "xyxyzz": boxes has format [xmin, ymin, xmax, ymax, zmin, zmax]
                "xywh": boxes has format [xmin, ymin, xsize, ysize]
                "xyzwhd": boxes has format [xmin, ymin, zmin, xsize, ysize, zsize]
                "ccwh": boxes has format [xcenter, ycenter, xsize, ysize]
                "cccwhd": boxes has format [xcenter, ycenter, zcenter, xsize, ysize, zsize]
            #. BoxMode class: choose from
                CornerCornerModeTypeA: equivalent to "xyxy" or "xyzxyz"
                CornerCornerModeTypeB: equivalent to "xxyy" or "xxyyzz"
                CornerCornerModeTypeC: equivalent to "xyxy" or "xyxyzz"
                CornerSizeMode: equivalent to "xywh" or "xyzwhd"
                CenterSizeMode: equivalent to "ccwh" or "cccwhd"
            #. BoxMode instance
            #. None: will assume mode is StandardMode
    Returns:
        boxes_standard: bounding box with standard mode, does not share memory with original boxes

    Example:
        boxes = torch.ones(10,6)
        box_convert_standard_mode(boxes=boxes, mode="xxyyzz")
    """
    return convert_box_mode(boxes=boxes, src_mode=mode, dst_mode=StandardMode())
