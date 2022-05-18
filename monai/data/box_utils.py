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

# We support 2-D or 3-D bounding boxes
SUPPORTED_SPATIAL_DIMS = [2, 3]

# We support the conversion between several box modes, i.e., representation of a bounding boxes
SUPPORTED_MODES = [CornerCornerModeTypeA, CornerCornerModeTypeB, CornerCornerModeTypeC, CornerSizeMode, CenterSizeMode]
# The standard box mode we use in all the box util functions
StandardMode = CornerCornerModeTypeA


def get_spatial_dims(
    boxes: Union[torch.Tensor, np.ndarray, None] = None,
    points: Union[torch.Tensor, np.ndarray, None] = None,
    corners: Union[Sequence, None] = None,
    spatial_size: Union[Sequence[int], torch.Tensor, np.ndarray, None] = None,
) -> int:
    """
    Get spatial dimension for the giving setting.
    Missing input is allowed. But at least one of the input value should be given.
    It raises ValueError if the dimensions of multiple inputs do not match with each other.

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray
        points: point coordinates, [x, y] or [x, y, z], Nx2 or Nx3 torch tensor or ndarray
        corners: corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch tensor or ndarray
        spatial_size: The spatial size of the image where the boxes are attached.
                len(spatial_size) should be in [2, 3].

    Returns:
        ``int``: spatial_dims, number of spatial dimensions of the bounding boxes.

    Example:
        .. code-block:: python

            boxes = torch.ones(10,6)
            get_spatial_dims(boxes, spatial_size=[100,200,200]) # will return 3
            get_spatial_dims(boxes, spatial_size=[100,200]) # will raise ValueError
            get_spatial_dims(boxes) # will return 3
    """
    spatial_dims_set = set()

    if boxes is not None:
        if int(boxes.shape[1]) not in [4, 6]:
            raise ValueError(
                f"Currently we support only boxes with shape [N,4] or [N,6], got boxes with shape {boxes.shape}."
            )
        spatial_dims_set.add(int(boxes.shape[1] / 2))
    if points is not None:
        if int(points.shape[1]) not in SUPPORTED_SPATIAL_DIMS:
            raise ValueError(
                f"Currently we support only points with shape [N,2] or [N,3], got boxes with shape {points.shape}."
            )
        spatial_dims_set.add(int(points.shape[1]))
    if corners is not None:
        if len(corners) not in [4, 6]:
            raise ValueError(
                f"Currently we support only boxes with shape [N,4] or [N,6], got box corner tuple with length {len(corners)}."
            )
        spatial_dims_set.add(len(corners) // 2)
    if spatial_size is not None:
        if len(spatial_size) not in SUPPORTED_SPATIAL_DIMS:
            raise ValueError(
                f"Currently we support only boxes on 2-D and 3-D images, got image spatial_size {spatial_size}."
            )
        spatial_dims_set.add(len(spatial_size))

    spatial_dims_list = list(spatial_dims_set)
    if len(spatial_dims_list) == 0:
        raise ValueError("At least one of the inputs needs to be non-empty.")
    elif len(spatial_dims_list) == 1:
        spatial_dims = int(spatial_dims_list[0])
        spatial_dims = look_up_option(spatial_dims, supported=[2, 3])
        return int(spatial_dims)
    else:
        raise ValueError("The dimensions of multiple inputs should match with each other.")


def get_boxmode(mode: Union[str, BoxMode, Type[BoxMode], None] = None, *args, **kwargs) -> BoxMode:
    """
    This function returns BoxMode object giving a representation of box mode

    Args:
        mode: a representation of box mode. If it is not given, this func will assume it is ``StandardMode``.

    Note:
        ``StandardMode`` is equivalent to :class:`~monai.data.box_mode.CornerCornerModeTypeA`.

        mode can be:
            #. str: choose from :class:`~monai.utils.enums.BoxModeName`, for example,
                - "xyxy": boxes has format [xmin, ymin, xmax, ymax]
                - "xyzxyz": boxes has format [xmin, ymin, zmin, xmax, ymax, zmax]
                - "xxyy": boxes has format [xmin, xmax, ymin, ymax]
                - "xxyyzz": boxes has format [xmin, xmax, ymin, ymax, zmin, zmax]
                - "xyxyzz": boxes has format [xmin, ymin, xmax, ymax, zmin, zmax]
                - "xywh": boxes has format [xmin, ymin, xsize, ysize]
                - "xyzwhd": boxes has format [xmin, ymin, zmin, xsize, ysize, zsize]
                - "ccwh": boxes has format [xcenter, ycenter, xsize, ysize]
                - "cccwhd": boxes has format [xcenter, ycenter, zcenter, xsize, ysize, zsize]
            #. BoxMode class: choose from the subclasses of :class:`~monai.data.box_mode.BoxMode`, for example,
                - CornerCornerModeTypeA: equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB: equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC: equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode: equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode: equivalent to "ccwh" or "cccwhd"
            #. BoxMode object: choose from the subclasses of :class:`~monai.data.box_mode.BoxMode`, for example,
                - CornerCornerModeTypeA(): equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB(): equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC(): equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode(): equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode(): equivalent to "ccwh" or "cccwhd"
            #. None: will assume mode is ``StandardMode``

    Returns:
        BoxMode object

    Example:
        .. code-block:: python

            mode = "xyzxyz"
            get_boxmode(mode) # will return CornerCornerModeTypeA()
    """
    if isinstance(mode, BoxMode):
        return mode

    boxmode: Type[BoxMode]
    if inspect.isclass(mode) and issubclass(mode, BoxMode):
        boxmode = mode
    elif isinstance(mode, str):
        for m in SUPPORTED_MODES:
            for n in SUPPORTED_SPATIAL_DIMS:
                if m.get_name(n) == mode:
                    boxmode = m
    elif mode is None:
        boxmode = StandardMode
    else:
        raise ValueError(f"Unsupported box mode: {mode}.")
    return boxmode(*args, **kwargs)


def _check_corners(corners: Sequence) -> bool:
    """
    Internal function to check the validity for the given box corners

    Args:
        corners: corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch tensor
        (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax)

    Returns:
        ``bool``: whether the box is valid

    Example:
        .. code-block:: python

            corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
            check_corner(corners) will return True
    """
    spatial_dims = get_spatial_dims(corners=corners)
    for axis in range(0, spatial_dims):
        if (corners[spatial_dims + axis] < corners[axis]).sum() > 0:
            return False
    return True


def convert_box_mode(
    boxes: NdarrayOrTensor,
    src_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
    dst_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
) -> NdarrayOrTensor:
    """
    This function converts the boxes in src_mode to the dst_mode.

    Args:
        boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray.
        src_mode: source box mode. If it is not given, this func will assume it is ``StandardMode``.
        dst_mode: target box mode. If it is not given, this func will assume it is ``StandardMode``.

    Note:
        ``StandardMode`` is equivalent to :class:`~monai.data.box_mode.CornerCornerModeTypeA`.

        ``src_mode`` and ``dst_mode`` can be:
            #. str: choose from :class:`~monai.utils.enums.BoxModeName`, for example,
                - "xyxy": boxes has format [xmin, ymin, xmax, ymax]
                - "xyzxyz": boxes has format [xmin, ymin, zmin, xmax, ymax, zmax]
                - "xxyy": boxes has format [xmin, xmax, ymin, ymax]
                - "xxyyzz": boxes has format [xmin, xmax, ymin, ymax, zmin, zmax]
                - "xyxyzz": boxes has format [xmin, ymin, xmax, ymax, zmin, zmax]
                - "xywh": boxes has format [xmin, ymin, xsize, ysize]
                - "xyzwhd": boxes has format [xmin, ymin, zmin, xsize, ysize, zsize]
                - "ccwh": boxes has format [xcenter, ycenter, xsize, ysize]
                - "cccwhd": boxes has format [xcenter, ycenter, zcenter, xsize, ysize, zsize]
            #. BoxMode class: choose from the subclasses of :class:`~monai.data.box_mode.BoxMode`, for example,
                - CornerCornerModeTypeA: equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB: equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC: equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode: equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode: equivalent to "ccwh" or "cccwhd"
            #. BoxMode object: choose from the subclasses of :class:`~monai.data.box_mode.BoxMode`, for example,
                - CornerCornerModeTypeA(): equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB(): equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC(): equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode(): equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode(): equivalent to "ccwh" or "cccwhd"
            #. None: will assume mode is ``StandardMode``

    Returns:
        bounding boxes with target mode, with same format as ``boxes``, does not share memory with ``boxes``

    Example:
        .. code-block:: python

            boxes = torch.ones(10,4)
            # The following three lines are equivalent
            # They convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
            box_convert_mode(boxes=boxes, src_mode="xyxy", dst_mode="ccwh")
            box_convert_mode(boxes=boxes, src_mode="xyxy", dst_mode=monai.data.box_mode.CenterSizeMode)
            box_convert_mode(boxes=boxes, src_mode="xyxy", dst_mode=monai.data.box_mode.CenterSizeMode())
    """
    src_boxmode = get_boxmode(src_mode)
    dst_boxmode = get_boxmode(dst_mode)

    # if mode not changed, deepcopy the original boxes
    if isinstance(src_boxmode, type(dst_boxmode)):
        return deepcopy(boxes)

    # convert box mode
    # convert numpy to tensor if needed
    boxes_t, *_ = convert_data_type(boxes, torch.Tensor)

    # convert boxes to corners
    corners = src_boxmode.boxes_to_corners(boxes_t)

    # check validity of corners
    if not _check_corners(corners):
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
    Convert given boxes to standard mode.
    Standard mode is "xyxy" or "xyzxyz",
    representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

    Args:
        boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray.
        mode: source box mode. If it is not given, this func will assume it is ``StandardMode``.
            It follows the same format with ``src_mode`` and ``dst_mode`` in :func:`~monai.data.box_utils.convert_box_mode`.

    Returns:
        bounding boxes with standard mode, with same format as ``boxes``, does not share memory with ``boxes``

    Example:
        .. code-block:: python

            boxes = torch.ones(10,6)
            # The following two lines are equivalent
            # They convert boxes with format [xmin, xmax, ymin, ymax, zmin, zmax] to [xmin, ymin, zmin, xmax, ymax, zmax]
            box_convert_standard_mode(boxes=boxes, mode="xxyyzz")
            box_convert_mode(boxes=boxes, src_mode="xxyyzz", dst_mode="xyzxyz")
    """
    return convert_box_mode(boxes=boxes, src_mode=mode, dst_mode=StandardMode())
