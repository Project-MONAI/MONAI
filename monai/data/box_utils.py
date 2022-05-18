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
This utility module mainly supports rectangular bounding boxes with a few
different parameterizations and methods for converting between them. It
provides reliable access to the spatial coordinates of the box vertices in the
"canonical ordering":
[xmin, ymin, xmax, ymax] for 2D and [xmin, ymin, zmin, xmax, ymax, zmax] for 3D.
We currently define this ordering as `monai.data.box_utils.StandardMode` and
the rest of the detection pipelines mainly assumes boxes in `StandardMode`.
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Sequence, Tuple, Type, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import look_up_option
from monai.utils.enums import BoxModeName
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

# We support 2-D or 3-D bounding boxes
SUPPORTED_SPATIAL_DIMS = [2, 3]


# TO_REMOVE = 0.0 if the bottom-right corner pixel/voxel is not included in the box,
#      i.e., when xmin=1., xmax=2., we have w = 1.
# TO_REMOVE = 1.0  if the bottom-right corner pixel/voxel is included in the box,
#       i.e., when xmin=1., xmax=2., we have w = 2.
# Currently, only `TO_REMOVE = 0.0` is supported
TO_REMOVE = 0.0  # xmax-xmin = w -TO_REMOVE.


class BoxMode(ABC):
    """
    An abstract class of a ``BoxMode``.

    A ``BoxMode`` is callable that converts box mode of ``boxes``, which are Nx4 (2D) or Nx6 (3D) torch tensor or ndarray.
    ``BoxMode`` has several subclasses that represents different box modes, including

    - :class:`~monai.data.box_utils.CornerCornerModeTypeA`:
      represents [xmin, ymin, xmax, ymax] for 2D and [xmin, ymin, zmin, xmax, ymax, zmax] for 3D
    - :class:`~monai.data.box_utils.CornerCornerModeTypeB`:
      represents [xmin, xmax, ymin, ymax] for 2D and [xmin, xmax, ymin, ymax, zmin, zmax] for 3D
    - :class:`~monai.data.box_utils.CornerCornerModeTypeC`:
      represents [xmin, ymin, xmax, ymax] for 2D and [xmin, ymin, xmax, ymax, zmin, zmax] for 3D
    - :class:`~monai.data.box_utils.CornerSizeMode`:
      represents [xmin, ymin, xsize, ysize] for 2D and [xmin, ymin, zmin, xsize, ysize, zsize] for 3D
    - :class:`~monai.data.box_utils.CenterSizeMode`:
      represents [xcenter, ycenter, xsize, ysize] for 2D and [xcenter, ycenter, zcenter, xsize, ysize, zsize] for 3D

    We currently define ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
    and monai detection pipelines mainly assume ``boxes`` are in ``StandardMode``.

    The implementation should be aware of:

    - remember to define class variable ``name``,
      a dictionary that maps ``spatial_dims`` to :class:`~monai.utils.enums.BoxModeName`.
    - :func:`~monai.data.box_utils.BoxMode.boxes_to_corners` and :func:`~monai.data.box_utils.BoxMode.corners_to_boxes`
      should not modify inputs in place.
    """

    # a dictionary that maps spatial_dims to monai.utils.enums.BoxModeName.
    name: Dict[int, BoxModeName] = {}

    @classmethod
    def get_name(cls, spatial_dims: int) -> str:
        """
        Get the mode name for the given spatial dimension using class variable ``name``.

        Args:
            spatial_dims: number of spatial dimensions of the bounding box.

        Returns:
            ``str``: mode string name
        """
        return cls.name[spatial_dims].value

    @abstractmethod
    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        """
        Convert the bounding boxes of the current mode to corners.

        Args:
            boxes: bounding box, Nx4 or Nx6 torch tensor

        Returns:
            ``Tuple``: corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch tensor.
            It represents (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax)

        Example:
            .. code-block:: python

                boxes = torch.ones(10,6)
                boxmode.boxes_to_corners(boxes) will return a 6-element tuple, each element is a 10x1 tensor
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        """
        Convert the given box corners to the bounding boxes of the current mode.

        Args:
            corners: corners of boxes, 4-element or 6-element tuple, each element is a Nx1 torch tensor.
                It represents (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax)

        Returns:
            ``Tensor``: bounding box, Nx4 or Nx6 torch tensor

        Example:
            .. code-block:: python

                corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
                boxmode.corners_to_boxes(corners) will return a 10x4 tensor
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class CornerCornerModeTypeA(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xyxy" or "xyzxyz", with format of
    [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

    Example:
        .. code-block:: python

            CornerCornerModeTypeA.get_name(spatial_dims=2) # will return "xyxy"
            CornerCornerModeTypeA.get_name(spatial_dims=3) # will return "xyzxyz"
    """

    name = {2: BoxModeName.XYXY, 3: BoxModeName.XYZXYZ}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        corners: Tuple
        corners = boxes.split(1, dim=-1)
        return corners

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        boxes: torch.Tensor
        boxes = torch.cat(tuple(corners), dim=-1)
        return boxes


class CornerCornerModeTypeB(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xxyy" or "xxyyzz", with format of
    [xmin, xmax, ymin, ymax] or [xmin, xmax, ymin, ymax, zmin, zmax].

    Example:
        .. code-block:: python

            CornerCornerModeTypeB.get_name(spatial_dims=2) # will return "xxyy"
            CornerCornerModeTypeB.get_name(spatial_dims=3) # will return "xxyyzz"
    """

    name = {2: BoxModeName.XXYY, 3: BoxModeName.XXYYZZ}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        corners: Tuple
        spatial_dims = get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = boxes.split(1, dim=-1)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xmin, xmax, ymin, ymax = boxes.split(1, dim=-1)
            corners = xmin, ymin, xmax, ymax
        return corners

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        boxes: torch.Tensor
        spatial_dims = get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            boxes = torch.cat((corners[0], corners[3], corners[1], corners[4], corners[2], corners[5]), dim=-1)
        elif spatial_dims == 2:
            boxes = torch.cat((corners[0], corners[2], corners[1], corners[3]), dim=-1)
        return boxes


class CornerCornerModeTypeC(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xyxy" or "xyxyzz", with format of
    [xmin, ymin, xmax, ymax] or [xmin, ymin, xmax, ymax, zmin, zmax].

    Example:
        .. code-block:: python

            CornerCornerModeTypeC.get_name(spatial_dims=2) # will return "xyxy"
            CornerCornerModeTypeC.get_name(spatial_dims=3) # will return "xyxyzz"
    """

    name = {2: BoxModeName.XYXY, 3: BoxModeName.XYXYZZ}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        corners: Tuple
        spatial_dims = get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xmin, ymin, xmax, ymax, zmin, zmax = boxes.split(1, dim=-1)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            corners = boxes.split(1, dim=-1)
        return corners

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        boxes: torch.Tensor
        spatial_dims = get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            boxes = torch.cat((corners[0], corners[1], corners[3], corners[4], corners[2], corners[5]), dim=-1)
        elif spatial_dims == 2:
            boxes = torch.cat(tuple(corners), dim=-1)
        return boxes


class CornerSizeMode(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xywh" or "xyzwhd", with format of
    [xmin, ymin, xsize, ysize] or [xmin, ymin, zmin, xsize, ysize, zsize].

    Example:
        .. code-block:: python

            CornerSizeMode.get_name(spatial_dims=2) # will return "xywh"
            CornerSizeMode.get_name(spatial_dims=3) # will return "xyzwhd"
    """

    name = {2: BoxModeName.XYWH, 3: BoxModeName.XYZWHD}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        corners: Tuple
        # convert to float32 when computing torch.clamp, which does not support float16
        box_dtype = boxes.dtype
        compute_dtype = torch.float32

        spatial_dims = get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xmin, ymin, zmin, w, h, d = boxes.split(1, dim=-1)
            xmax = xmin + (w - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = ymin + (h - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            zmax = zmin + (d - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xmin, ymin, w, h = boxes.split(1, dim=-1)
            xmax = xmin + (w - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = ymin + (h - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            corners = xmin, ymin, xmax, ymax
        return corners

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        boxes: torch.Tensor
        spatial_dims = get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            xmin, ymin, zmin, xmax, ymax, zmax = corners[0], corners[1], corners[2], corners[3], corners[4], corners[5]
            boxes = torch.cat(
                (xmin, ymin, zmin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE, zmax - zmin + TO_REMOVE), dim=-1
            )
        elif spatial_dims == 2:
            xmin, ymin, xmax, ymax = corners[0], corners[1], corners[2], corners[3]
            boxes = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)
        return boxes


class CenterSizeMode(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "ccwh" or "cccwhd", with format of
    [xmin, ymin, xsize, ysize] or [xmin, ymin, zmin, xsize, ysize, zsize].

    Example:
        .. code-block:: python

            CenterSizeMode.get_name(spatial_dims=2) # will return "ccwh"
            CenterSizeMode.get_name(spatial_dims=3) # will return "cccwhd"
    """

    name = {2: BoxModeName.CCWH, 3: BoxModeName.CCCWHD}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        corners: Tuple
        # convert to float32 when computing torch.clamp, which does not support float16
        box_dtype = boxes.dtype
        compute_dtype = torch.float32

        spatial_dims = get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xc, yc, zc, w, h, d = boxes.split(1, dim=-1)
            xmin = xc - ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            xmax = xc + ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymin = yc - ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = yc + ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            zmin = zc - ((d - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            zmax = zc + ((d - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xc, yc, w, h = boxes.split(1, dim=-1)
            xmin = xc - ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            xmax = xc + ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymin = yc - ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = yc + ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            corners = xmin, ymin, xmax, ymax
        return corners

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        boxes: torch.Tensor
        spatial_dims = get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            xmin, ymin, zmin, xmax, ymax, zmax = corners[0], corners[1], corners[2], corners[3], corners[4], corners[5]
            boxes = torch.cat(
                (
                    (xmin + xmax + TO_REMOVE) / 2.0,
                    (ymin + ymax + TO_REMOVE) / 2.0,
                    (zmin + zmax + TO_REMOVE) / 2.0,
                    xmax - xmin + TO_REMOVE,
                    ymax - ymin + TO_REMOVE,
                    zmax - zmin + TO_REMOVE,
                ),
                dim=-1,
            )
        elif spatial_dims == 2:
            xmin, ymin, xmax, ymax = corners[0], corners[1], corners[2], corners[3]
            boxes = torch.cat(
                (
                    (xmin + xmax + TO_REMOVE) / 2.0,
                    (ymin + ymax + TO_REMOVE) / 2.0,
                    xmax - xmin + TO_REMOVE,
                    ymax - ymin + TO_REMOVE,
                ),
                dim=-1,
            )
        return boxes


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
    Get spatial dimension for the giving setting and check the validity of them.
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

    # Check the validity of each input and add its corresponding spatial_dims to spatial_dims_set
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

    # Get spatial_dims from spatial_dims_set, which contains only unique values
    spatial_dims_list = list(spatial_dims_set)
    if len(spatial_dims_list) == 0:
        raise ValueError("At least one of the inputs needs to be non-empty.")

    if len(spatial_dims_list) == 1:
        spatial_dims = int(spatial_dims_list[0])
        spatial_dims = look_up_option(spatial_dims, supported=[2, 3])
        return int(spatial_dims)

    raise ValueError("The dimensions of multiple inputs should match with each other.")


def get_boxmode(mode: Union[str, BoxMode, Type[BoxMode], None] = None, *args, **kwargs) -> BoxMode:
    """
    This function that return a :class:`~monai.data.box_utils.BoxMode` object giving a representation of box mode

    Args:
        mode: a representation of box mode. If it is not given, this func will assume it is ``StandardMode()``.

    Note:
        ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
        also represented as "xyxy" for 2D and "xyzxyz" for 3D.

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
            #. BoxMode class: choose from the subclasses of :class:`~monai.data.box_utils.BoxMode`, for example,
                - CornerCornerModeTypeA: equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB: equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC: equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode: equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode: equivalent to "ccwh" or "cccwhd"
            #. BoxMode object: choose from the subclasses of :class:`~monai.data.box_utils.BoxMode`, for example,
                - CornerCornerModeTypeA(): equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB(): equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC(): equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode(): equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode(): equivalent to "ccwh" or "cccwhd"
            #. None: will assume mode is ``StandardMode()``

    Returns:
        BoxMode object

    Example:
        .. code-block:: python

            mode = "xyzxyz"
            get_boxmode(mode) # will return CornerCornerModeTypeA()
    """
    if isinstance(mode, BoxMode):
        return mode

    if inspect.isclass(mode) and issubclass(mode, BoxMode):
        return mode(*args, **kwargs)

    if isinstance(mode, str):
        for m in SUPPORTED_MODES:
            for n in SUPPORTED_SPATIAL_DIMS:
                if inspect.isclass(m) and issubclass(m, BoxMode) and m.get_name(n) == mode:
                    return m(*args, **kwargs)

    if mode is not None:
        raise ValueError(f"Unsupported box mode: {mode}.")
    return StandardMode(*args, **kwargs)


def convert_box_mode(
    boxes: NdarrayOrTensor,
    src_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
    dst_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
) -> NdarrayOrTensor:
    """
    This function converts the boxes in src_mode to the dst_mode.

    Args:
        boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray.
        src_mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
            It follows the same format with ``mode`` in :func:`~monai.data.box_utils.get_boxmode`.
        dst_mode: target box mode. If it is not given, this func will assume it is ``StandardMode()``.
            It follows the same format with ``mode`` in :func:`~monai.data.box_utils.get_boxmode`.

    Returns:
        bounding boxes with target mode, with same data type as ``boxes``, does not share memory with ``boxes``

    Example:
        .. code-block:: python

            boxes = torch.ones(10,4)
            # The following three lines are equivalent
            # They convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
            box_convert_mode(boxes=boxes, src_mode="xyxy", dst_mode="ccwh")
            box_convert_mode(boxes=boxes, src_mode="xyxy", dst_mode=monai.data.box_utils.CenterSizeMode)
            box_convert_mode(boxes=boxes, src_mode="xyxy", dst_mode=monai.data.box_utils.CenterSizeMode())
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
    spatial_dims = get_spatial_dims(boxes=boxes_t)
    for axis in range(0, spatial_dims):
        if (corners[spatial_dims + axis] < corners[axis]).sum() > 0:
            warnings.warn("Given boxes has invalid values. The box size must be non-negative.")

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
        mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
            It follows the same format with ``mode`` in :func:`~monai.data.box_utils.get_boxmode`.

    Returns:
        bounding boxes with standard mode, with same data type as ``boxes``, does not share memory with ``boxes``

    Example:
        .. code-block:: python

            boxes = torch.ones(10,6)
            # The following two lines are equivalent
            # They convert boxes with format [xmin, xmax, ymin, ymax, zmin, zmax] to [xmin, ymin, zmin, xmax, ymax, zmax]
            box_convert_standard_mode(boxes=boxes, mode="xxyyzz")
            box_convert_mode(boxes=boxes, src_mode="xxyyzz", dst_mode="xyzxyz")
    """
    return convert_box_mode(boxes=boxes, src_mode=mode, dst_mode=StandardMode())
