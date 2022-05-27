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
from typing import Callable, Dict, Sequence, Tuple, Type, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import look_up_option
from monai.utils.enums import BoxModeName
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

# We support 2-D or 3-D bounding boxes
SUPPORTED_SPATIAL_DIMS = [2, 3]


# TO_REMOVE = 0.0 if the bottom-right corner pixel/voxel is not included in the boxes,
#      i.e., when xmin=1., xmax=2., we have w = 1.
# TO_REMOVE = 1.0  if the bottom-right corner pixel/voxel is included in the boxes,
#       i.e., when xmin=1., xmax=2., we have w = 2.
# Currently, only `TO_REMOVE = 0.0` is supported
TO_REMOVE = 0.0  # xmax-xmin = w -TO_REMOVE.

# Some torch functions do not support half precision.
# We therefore compute those functions under COMPUTE_DTYPE
COMPUTE_DTYPE = torch.float32


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
            spatial_dims: number of spatial dimensions of the bounding boxes.

        Returns:
            ``str``: mode string name
        """
        return cls.name[spatial_dims].value

    @abstractmethod
    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        """
        Convert the bounding boxes of the current mode to corners.

        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor

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
            ``Tensor``: bounding boxes, Nx4 or Nx6 torch tensor

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

        spatial_dims = get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xmin, ymin, zmin, w, h, d = boxes.split(1, dim=-1)
            xmax = xmin + (w - TO_REMOVE).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            ymax = ymin + (h - TO_REMOVE).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            zmax = zmin + (d - TO_REMOVE).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xmin, ymin, w, h = boxes.split(1, dim=-1)
            xmax = xmin + (w - TO_REMOVE).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            ymax = ymin + (h - TO_REMOVE).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
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

        spatial_dims = get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xc, yc, zc, w, h, d = boxes.split(1, dim=-1)
            xmin = xc - ((w - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            xmax = xc + ((w - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            ymin = yc - ((h - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            ymax = yc + ((h - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            zmin = zc - ((d - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            zmax = zc + ((d - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xc, yc, w, h = boxes.split(1, dim=-1)
            xmin = xc - ((w - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            xmax = xc + ((w - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            ymin = yc - ((h - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
            ymax = yc + ((h - TO_REMOVE) / 2.0).to(dtype=COMPUTE_DTYPE).clamp(min=0).to(dtype=box_dtype)
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


def box_centers(boxes: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    Compute center points of boxes

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

    Returns:
        center points with size of (N, spatial_dims)

    """
    spatial_dims = get_spatial_dims(boxes=boxes)
    return convert_box_mode(boxes=boxes, src_mode=StandardMode, dst_mode=CenterSizeMode)[:, :spatial_dims]


def centers_in_boxes(centers: NdarrayOrTensor, boxes: NdarrayOrTensor, eps: float = 0.01) -> NdarrayOrTensor:
    """
    Checks which center points are within boxes

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``.
        centers: center points, Nx2 or Nx3 torch tensor or ndarray.
        eps: minimum distance to border of boxes.

    Returns:
        boolean array indicating which center points are within the boxes, sized (N,).

    Reference:
        https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/ops.py

    """
    spatial_dims = get_spatial_dims(boxes=boxes)

    # compute relative position of centers compared to borders
    # should be non-negative if centers are within boxes
    center_to_border = [centers[:, axis] - boxes[:, axis] for axis in range(spatial_dims)] + [
        boxes[:, axis + spatial_dims] - centers[:, axis] for axis in range(spatial_dims)
    ]

    if isinstance(boxes, np.ndarray):
        min_center_to_border: np.ndarray = np.stack(center_to_border, axis=1).min(axis=1)
        return min_center_to_border > eps  # array[bool]

    return torch.stack(center_to_border, dim=1).to(COMPUTE_DTYPE).min(dim=1)[0] > eps  # Tensor[bool]


def boxes_center_distance(
    boxes1: NdarrayOrTensor, boxes2: NdarrayOrTensor, euclidean: bool = True
) -> Tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor]:
    """
    Distance of center points between two sets of boxes

    Args:
        boxes1: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        boxes2: bounding boxes, Mx4 or Mx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        euclidean: computed the euclidean distance otherwise it uses the l1 distance

    Returns:
        - The pairwise distances for every element in boxes1 and boxes2,
          with size of (N,M) and same data type as ``boxes1``.
        - Center points of boxes1, with size of (N,spatial_dims) and same data type as ``boxes1``.
        - Center points of boxes2, with size of (M,spatial_dims) and same data type as ``boxes1``.

    Reference:
        https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/ops.py

    """

    if not isinstance(boxes1, type(boxes2)):
        warnings.warn(f"boxes1 is {type(boxes1)}, while boxes2 is {type(boxes2)}. The result will be {type(boxes1)}.")

    # convert numpy to tensor if needed
    boxes1_t, *_ = convert_data_type(boxes1, torch.Tensor)
    boxes2_t, *_ = convert_data_type(boxes2, torch.Tensor)

    center1 = box_centers(boxes1_t.to(COMPUTE_DTYPE))  # (N, spatial_dims)
    center2 = box_centers(boxes2_t.to(COMPUTE_DTYPE))  # (M, spatial_dims)

    if euclidean:
        dists = (center1[:, None] - center2[None]).pow(2).sum(-1).sqrt()
    else:
        # before sum: (N, M, spatial_dims)
        dists = (center1[:, None] - center2[None]).sum(-1)

    # convert tensor back to numpy if needed
    (dists, center1, center2), *_ = convert_to_dst_type(src=(dists, center1, center2), dst=boxes1)
    return dists, center1, center2


def is_valid_box_values(boxes: NdarrayOrTensor) -> bool:
    """
    This function checks whether the box size is non-negative.

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

    Returns:
        whether ``boxes`` is valid
    """
    spatial_dims = get_spatial_dims(boxes=boxes)
    for axis in range(0, spatial_dims):
        if (boxes[:, spatial_dims + axis] < boxes[:, axis]).sum() > 0:
            return False
    return True


def box_area(boxes: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    This function computes the area (2D) or volume (3D) of each box.
    Half precision is not recommended for this function as it may cause overflow, especially for 3D images.

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

    Returns:
        area (2D) or volume (3D) of boxes, with size of (N,).

    Example:
        .. code-block:: python

            boxes = torch.ones(10,6)
            # we do computation with torch.float32 to avoid overflow
            compute_dtype = torch.float32
            area = box_area(boxes=boxes.to(dtype=compute_dtype))  # torch.float32, size of (10,)
    """

    if not is_valid_box_values(boxes):
        raise ValueError("Given boxes has invalid values. The box size must be non-negative.")

    spatial_dims = get_spatial_dims(boxes=boxes)

    area = boxes[:, spatial_dims] - boxes[:, 0] + TO_REMOVE
    for axis in range(1, spatial_dims):
        area = area * (boxes[:, axis + spatial_dims] - boxes[:, axis] + TO_REMOVE)

    # convert numpy to tensor if needed
    area_t, *_ = convert_data_type(area, torch.Tensor)

    # check if NaN or Inf, especially for half precision
    if area_t.isnan().any() or area_t.isinf().any():
        if area_t.dtype is torch.float16:
            raise ValueError("Box area is NaN or Inf. boxes is float16. Please change to float32 and test it again.")
        else:
            raise ValueError("Box area is NaN or Inf.")

    return area


def _box_inter_union(
    boxes1_t: torch.Tensor, boxes2_t: torch.Tensor, compute_dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This internal function computes the intersection and union area of two set of boxes.

    Args:
        boxes1: bounding boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
        boxes2: bounding boxes, Mx4 or Mx6 torch tensor. The box mode is assumed to be ``StandardMode``
        compute_dtype: default torch.float32, dtype with which the results will be computed

    Returns:
        inter, with size of (N,M) and dtype of ``compute_dtype``.
        union, with size of (N,M) and dtype of ``compute_dtype``.

    """
    spatial_dims = get_spatial_dims(boxes=boxes1_t)

    # compute area with float32
    area1 = box_area(boxes=boxes1_t.to(dtype=compute_dtype))  # (N,)
    area2 = box_area(boxes=boxes2_t.to(dtype=compute_dtype))  # (M,)

    # get the left top and right bottom points for the NxM combinations
    lt = torch.max(boxes1_t[:, None, :spatial_dims], boxes2_t[:, :spatial_dims]).to(
        dtype=compute_dtype
    )  # (N,M,spatial_dims) left top
    rb = torch.min(boxes1_t[:, None, spatial_dims:], boxes2_t[:, spatial_dims:]).to(
        dtype=compute_dtype
    )  # (N,M,spatial_dims) right bottom

    # compute size for the intersection region for the NxM combinations
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # (N,M,spatial_dims)
    inter = torch.prod(wh, dim=-1, keepdim=False)  # (N,M)

    union = area1[:, None] + area2 - inter
    return inter, union


def box_iou(boxes1: NdarrayOrTensor, boxes2: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    Compute the intersection over union (IoU) of two set of boxes.

    Args:
        boxes1: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        boxes2: bounding boxes, Mx4 or Mx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

    Returns:
        IoU, with size of (N,M) and same data type as ``boxes1``

    """

    if not isinstance(boxes1, type(boxes2)):
        warnings.warn(f"boxes1 is {type(boxes1)}, while boxes2 is {type(boxes2)}. The result will be {type(boxes1)}.")

    # convert numpy to tensor if needed
    boxes1_t, *_ = convert_data_type(boxes1, torch.Tensor)
    boxes2_t, *_ = convert_data_type(boxes2, torch.Tensor)

    # we do computation with compute_dtype to avoid overflow
    box_dtype = boxes1_t.dtype

    inter, union = _box_inter_union(boxes1_t, boxes2_t, compute_dtype=COMPUTE_DTYPE)

    # compute IoU and convert back to original box_dtype
    iou_t = inter / (union + torch.finfo(COMPUTE_DTYPE).eps)  # (N,M)
    iou_t = iou_t.to(dtype=box_dtype)

    # check if NaN or Inf
    if torch.isnan(iou_t).any() or torch.isinf(iou_t).any():
        raise ValueError("Box IoU is NaN or Inf.")

    # convert tensor back to numpy if needed
    iou, *_ = convert_to_dst_type(src=iou_t, dst=boxes1)
    return iou


def box_giou(boxes1: NdarrayOrTensor, boxes2: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    Compute the generalized intersection over union (GIoU) of two sets of boxes.
    The two inputs can have different shapes and the func return an NxM matrix,
    (in contrary to :func:`~monai.data.box_utils.box_pair_giou` , which requires the inputs to have the same
    shape and returns ``N`` values).

    Args:
        boxes1: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        boxes2: bounding boxes, Mx4 or Mx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

    Returns:
        GIoU, with size of (N,M) and same data type as ``boxes1``

    Reference:
        https://giou.stanford.edu/GIoU.pdf

    """

    if not isinstance(boxes1, type(boxes2)):
        warnings.warn(f"boxes1 is {type(boxes1)}, while boxes2 is {type(boxes2)}. The result will be {type(boxes1)}.")

    # convert numpy to tensor if needed
    boxes1_t, *_ = convert_data_type(boxes1, torch.Tensor)
    boxes2_t, *_ = convert_data_type(boxes2, torch.Tensor)

    spatial_dims = get_spatial_dims(boxes=boxes1_t)

    # we do computation with compute_dtype to avoid overflow
    box_dtype = boxes1_t.dtype

    inter, union = _box_inter_union(boxes1_t, boxes2_t, compute_dtype=COMPUTE_DTYPE)
    iou = inter / (union + torch.finfo(COMPUTE_DTYPE).eps)  # (N,M)

    # Enclosure
    # get the left top and right bottom points for the NxM combinations
    lt = torch.min(boxes1_t[:, None, :spatial_dims], boxes2_t[:, :spatial_dims]).to(
        dtype=COMPUTE_DTYPE
    )  # (N,M,spatial_dims) left top
    rb = torch.max(boxes1_t[:, None, spatial_dims:], boxes2_t[:, spatial_dims:]).to(
        dtype=COMPUTE_DTYPE
    )  # (N,M,spatial_dims) right bottom

    # compute size for the enclosure region for the NxM combinations
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # (N,M,spatial_dims)
    enclosure = torch.prod(wh, dim=-1, keepdim=False)  # (N,M)

    # GIoU
    giou_t = iou - (enclosure - union) / (enclosure + torch.finfo(COMPUTE_DTYPE).eps)
    giou_t = giou_t.to(dtype=box_dtype)
    if torch.isnan(giou_t).any() or torch.isinf(giou_t).any():
        raise ValueError("Box GIoU is NaN or Inf.")

    # convert tensor back to numpy if needed
    giou, *_ = convert_to_dst_type(src=giou_t, dst=boxes1)
    return giou


def box_pair_giou(boxes1: NdarrayOrTensor, boxes2: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    Compute the generalized intersection over union (GIoU) of a pair of boxes.
    The two inputs should have the same shape and the func return an (N,) array,
    (in contrary to :func:`~monai.data.box_utils.box_giou` , which does not require the inputs to have the same
    shape and returns ``NxM`` matrix).

    Args:
        boxes1: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        boxes2: bounding boxes, same shape with boxes1. The box mode is assumed to be ``StandardMode``

    Returns:
        paired GIoU, with size of (N,) and same data type as ``boxes1``

    Reference:
        https://giou.stanford.edu/GIoU.pdf

    """

    if not isinstance(boxes1, type(boxes2)):
        warnings.warn(f"boxes1 is {type(boxes1)}, while boxes2 is {type(boxes2)}. The result will be {type(boxes1)}.")

    # convert numpy to tensor if needed
    boxes1_t, *_ = convert_data_type(boxes1, torch.Tensor)
    boxes2_t, *_ = convert_data_type(boxes2, torch.Tensor)

    if boxes1_t.shape != boxes2_t.shape:
        raise ValueError("boxes1 and boxes2 should be paired and have same shape.")

    spatial_dims = get_spatial_dims(boxes=boxes1_t)

    # we do computation with compute_dtype to avoid overflow
    box_dtype = boxes1_t.dtype

    # compute area
    area1 = box_area(boxes=boxes1_t.to(dtype=COMPUTE_DTYPE))  # (N,)
    area2 = box_area(boxes=boxes2_t.to(dtype=COMPUTE_DTYPE))  # (N,)

    # Intersection
    # get the left top and right bottom points for the boxes pair
    lt = torch.max(boxes1_t[:, :spatial_dims], boxes2_t[:, :spatial_dims]).to(
        dtype=COMPUTE_DTYPE
    )  # (N,spatial_dims) left top
    rb = torch.min(boxes1_t[:, spatial_dims:], boxes2_t[:, spatial_dims:]).to(
        dtype=COMPUTE_DTYPE
    )  # (N,spatial_dims) right bottom

    # compute size for the intersection region for the boxes pair
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # (N,spatial_dims)
    inter = torch.prod(wh, dim=-1, keepdim=False)  # (N,)

    # compute IoU and convert back to original box_dtype
    union = area1 + area2 - inter
    iou = inter / (union + torch.finfo(COMPUTE_DTYPE).eps)  # (N,)

    # Enclosure
    # get the left top and right bottom points for the boxes pair
    lt = torch.min(boxes1_t[:, :spatial_dims], boxes2_t[:, :spatial_dims]).to(
        dtype=COMPUTE_DTYPE
    )  # (N,spatial_dims) left top
    rb = torch.max(boxes1_t[:, spatial_dims:], boxes2_t[:, spatial_dims:]).to(
        dtype=COMPUTE_DTYPE
    )  # (N,spatial_dims) right bottom

    # compute size for the enclose region for the boxes pair
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # (N,spatial_dims)
    enclosure = torch.prod(wh, dim=-1, keepdim=False)  # (N,)

    giou_t = iou - (enclosure - union) / (enclosure + torch.finfo(COMPUTE_DTYPE).eps)
    giou_t = giou_t.to(dtype=box_dtype)  # (N,spatial_dims)
    if torch.isnan(giou_t).any() or torch.isinf(giou_t).any():
        raise ValueError("Box GIoU is NaN or Inf.")

    # convert tensor back to numpy if needed
    giou, *_ = convert_to_dst_type(src=giou_t, dst=boxes1)
    return giou


def spatial_crop_boxes(
    boxes: NdarrayOrTensor,
    roi_start: Union[Sequence[int], NdarrayOrTensor],
    roi_end: Union[Sequence[int], NdarrayOrTensor],
    remove_empty: bool = True,
) -> Tuple[NdarrayOrTensor, NdarrayOrTensor]:
    """
    This function generate the new boxes when the corresponding image is cropped to the given ROI.
    When ``remove_empty=True``, it makes sure the bounding boxes are within the new cropped image.

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        roi_start: voxel coordinates for start of the crop ROI, negative values allowed.
        roi_end: voxel coordinates for end of the crop ROI, negative values allowed.
        remove_empty: whether to remove the boxes that are actually empty

    Returns:
        - cropped boxes, boxes[keep], does not share memory with original boxes
        - ``keep``, it indicates whether each box in ``boxes`` are kept when ``remove_empty=True``.
    """

    roi_start_torch, *_ = convert_data_type(
        data=roi_start, output_type=torch.Tensor, dtype=torch.int16, wrap_sequence=True
    )
    roi_end_torch, *_ = convert_to_dst_type(src=roi_end, dst=roi_start_torch, wrap_sequence=True)
    roi_end_torch = torch.maximum(roi_end_torch, roi_start_torch)

    # convert numpy to tensor if needed
    boxes_t, *_ = convert_data_type(deepcopy(boxes), torch.Tensor)

    # convert to float32 since torch.clamp_ does not support float16
    boxes_t = boxes_t.to(dtype=COMPUTE_DTYPE)

    # makes sure the bounding boxes are within the patch
    spatial_dims = get_spatial_dims(boxes=boxes, spatial_size=roi_end)
    for axis in range(0, spatial_dims):
        boxes_t[:, axis].clamp_(min=roi_start_torch[axis], max=roi_end_torch[axis] - TO_REMOVE)
        boxes_t[:, axis + spatial_dims].clamp_(min=roi_start_torch[axis], max=roi_end_torch[axis] - TO_REMOVE)
        boxes_t[:, axis] -= roi_start_torch[axis]
        boxes_t[:, axis + spatial_dims] -= roi_start_torch[axis]

    # remove the boxes that are actually empty
    if remove_empty:
        keep_t = boxes_t[:, spatial_dims] >= boxes_t[:, 0] + 1 - TO_REMOVE
        for axis in range(1, spatial_dims):
            keep_t = keep_t & (boxes_t[:, axis + spatial_dims] >= boxes_t[:, axis] + 1 - TO_REMOVE)
        boxes_t = boxes_t[keep_t]

    # convert tensor back to numpy if needed
    boxes_keep, *_ = convert_to_dst_type(src=boxes_t, dst=boxes)
    keep, *_ = convert_to_dst_type(src=keep_t, dst=boxes, dtype=keep_t.dtype)

    return boxes_keep, keep


def clip_boxes_to_image(
    boxes: NdarrayOrTensor, spatial_size: Union[Sequence[int], NdarrayOrTensor], remove_empty: bool = True
) -> Tuple[NdarrayOrTensor, NdarrayOrTensor]:
    """
    This function clips the ``boxes`` to makes sure the bounding boxes are within the image.

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        spatial_size: The spatial size of the image where the boxes are attached. len(spatial_size) should be in [2, 3].
        remove_empty: whether to remove the boxes that are actually empty

    Returns:
        - clipped boxes, boxes[keep], does not share memory with original boxes
        - ``keep``, it indicates whether each box in ``boxes`` are kept when ``remove_empty=True``.
    """
    spatial_dims = get_spatial_dims(boxes=boxes, spatial_size=spatial_size)
    return spatial_crop_boxes(boxes, roi_start=[0] * spatial_dims, roi_end=spatial_size, remove_empty=remove_empty)


def non_max_suppression(
    boxes: NdarrayOrTensor,
    scores: NdarrayOrTensor,
    nms_thresh: float,
    max_proposals: int = -1,
    box_overlap_metric: Callable = box_iou,
) -> NdarrayOrTensor:
    """
    Non-maximum suppression (NMS).

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        scores: prediction scores of the boxes, sized (N,). This function keeps boxes with higher scores.
        nms_thresh: threshold of NMS. For boxes with overlap more than nms_thresh,
            we only keep the one with the highest score.
        max_proposals: maximum number of boxes it keeps.
            If ``max_proposals`` = -1, there is no limit on the number of boxes that are kept.
        box_overlap_metric: the metric to compute overlap between boxes.

    Returns:
        Indexes of ``boxes`` that are kept after NMS.

    Example:
        keep = non_max_suppression(boxes, scores, num_thresh=0.1)
        boxes_after_nms = boxes[keep]
    """

    # returns empty array if boxes is empty
    if boxes.shape[0] == 0:
        return convert_to_dst_type(src=np.array([]), dst=boxes)[0]

    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(
            f"boxes and scores should have same length, got boxes shape {boxes.shape}, scores shape {scores.shape}"
        )

    # convert tensor to numpy if needed
    boxes_t, *_ = convert_data_type(boxes, torch.Tensor)
    scores_t, *_ = convert_to_dst_type(scores, boxes_t)

    # sort boxes in desending order according to the scores
    sort_idxs = torch.argsort(scores_t, dim=0, descending=True)
    boxes_sort = deepcopy(boxes_t)[sort_idxs, :]

    # initialize the list of picked indexes
    pick = []
    idxs = torch.Tensor(list(range(0, boxes_sort.shape[0]))).to(torch.long)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # pick the first index in the indexes list and add the index value to the list of picked indexes
        i = int(idxs[0].item())
        pick.append(i)
        if len(pick) >= max_proposals >= 1:
            break

        # compute the IoU between the rest of the boxes and the box just picked
        box_overlap = box_overlap_metric(boxes_sort[idxs, :], boxes_sort[i : i + 1, :])

        # keep only indexes from the index list that have overlap < nms_thresh
        to_keep_idx = (box_overlap <= nms_thresh).flatten()
        to_keep_idx[0] = False  # always remove idxs[0]
        idxs = idxs[to_keep_idx]

    # return only the bounding boxes that were picked using the integer data type
    pick_idx = sort_idxs[pick]

    # convert numpy back to tensor if needed
    return convert_to_dst_type(src=pick_idx, dst=boxes, dtype=pick_idx.dtype)[0]
