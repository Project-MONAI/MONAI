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
from typing import Dict, Sequence, Tuple

import torch

import monai
from monai.utils.enums import BoxModeName

# TO_REMOVE = 0.0 if the bottom-right corner pixel/voxel is not included in the box,
#      i.e., when xmin=1., xmax=2., we have w = 1.
# TO_REMOVE = 1.0  if the bottom-right corner pixel/voxel is included in the box,
#       i.e., when xmin=1., xmax=2., we have w = 2.
# Currently, only `TO_REMOVE = 0.0` is supported
TO_REMOVE = 0.0  # xmax-xmin = w -TO_REMOVE.


class BoxMode(ABC):
    """
    An abstract class of a ``BoxMode``.
    A BoxMode is callable that converts box mode of boxes.
    It always creates a copy and will not modify boxes in place.

    The implementation should be aware of:
    remember to define class variable ``name`` which is a dictionary that maps ``spatial_dims`` to the box mode name.
    """

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

    Note:
        .. code-block:: python

            CornerCornerModeTypeA.get_name(spatial_dims=2) # will return "xyxy"
            CornerCornerModeTypeA.get_name(spatial_dims=3) # will return "xyzxyz"
    """

    name = {2: BoxModeName.XYXY, 3: BoxModeName.XYZXYZ}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        return boxes.split(1, dim=-1)

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        return torch.cat(corners, dim=-1)


class CornerCornerModeTypeB(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xxyy" or "xxyyzz", with format of
    [xmin, xmax, ymin, ymax] or [xmin, xmax, ymin, ymax, zmin, zmax].

    Note:
        .. code-block:: python

            CornerCornerModeTypeB.get_name(spatial_dims=2) # will return "xxyy"
            CornerCornerModeTypeB.get_name(spatial_dims=3) # will return "xxyyzz"
    """

    name = {2: BoxModeName.XXYY, 3: BoxModeName.XXYYZZ}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        spatial_dims = monai.data.box_utils.get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = boxes.split(1, dim=-1)
            return xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xmin, xmax, ymin, ymax = boxes.split(1, dim=-1)
            return xmin, ymin, xmax, ymax

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        spatial_dims = monai.data.box_utils.get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            return torch.cat((corners[0], corners[3], corners[1], corners[4], corners[2], corners[5]), dim=-1)
        elif spatial_dims == 2:
            return torch.cat((corners[0], corners[2], corners[1], corners[3]), dim=-1)


class CornerCornerModeTypeC(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xyxy" or "xyxyzz", with format of
    [xmin, ymin, xmax, ymax] or [xmin, ymin, xmax, ymax, zmin, zmax].

    Note:
        .. code-block:: python

            CornerCornerModeTypeC.get_name(spatial_dims=2) # will return "xyxy"
            CornerCornerModeTypeC.get_name(spatial_dims=3) # will return "xyxyzz"
    """

    name = {2: BoxModeName.XYXY, 3: BoxModeName.XYXYZZ}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        spatial_dims = monai.data.box_utils.get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xmin, ymin, xmax, ymax, zmin, zmax = boxes.split(1, dim=-1)
            return xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            return boxes.split(1, dim=-1)

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        spatial_dims = monai.data.box_utils.get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            return torch.cat((corners[0], corners[1], corners[3], corners[4], corners[2], corners[5]), dim=-1)
        elif spatial_dims == 2:
            return torch.cat(corners, dim=-1)


class CornerSizeMode(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "xywh" or "xyzwhd", with format of
    [xmin, ymin, xsize, ysize] or [xmin, ymin, zmin, xsize, ysize, zsize].

    Note:
        .. code-block:: python

            CornerSizeMode.get_name(spatial_dims=2) # will return "xywh"
            CornerSizeMode.get_name(spatial_dims=3) # will return "xyzwhd"
    """

    name = {2: BoxModeName.XYWH, 3: BoxModeName.XYZWHD}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        # convert to float32 when computing torch.clamp, which does not support float16
        box_dtype = boxes.dtype
        compute_dtype = torch.float32

        spatial_dims = monai.data.box_utils.get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xmin, ymin, zmin, w, h, d = boxes.split(1, dim=-1)
            xmax = xmin + (w - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = ymin + (h - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            zmax = zmin + (d - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            return xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xmin, ymin, w, h = boxes.split(1, dim=-1)
            xmax = xmin + (w - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = ymin + (h - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            return xmin, ymin, xmax, ymax

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        spatial_dims = monai.data.box_utils.get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            xmin, ymin, zmin, xmax, ymax, zmax = corners[0], corners[1], corners[2], corners[3], corners[4], corners[5]
            return torch.cat(
                (xmin, ymin, zmin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE, zmax - zmin + TO_REMOVE), dim=-1
            )
        elif spatial_dims == 2:
            xmin, ymin, xmax, ymax = corners[0], corners[1], corners[2], corners[3]
            return torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)


class CenterSizeMode(BoxMode):
    """
    A subclass of ``BoxMode``.

    Also represented as "ccwh" or "cccwhd", with format of
    [xmin, ymin, xsize, ysize] or [xmin, ymin, zmin, xsize, ysize, zsize].

    Note:
        .. code-block:: python

            CenterSizeMode.get_name(spatial_dims=2) # will return "ccwh"
            CenterSizeMode.get_name(spatial_dims=3) # will return "cccwhd"
    """

    name = {2: BoxModeName.CCWH, 3: BoxModeName.CCCWHD}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        # convert to float32 when computing torch.clamp, which does not support float16
        box_dtype = boxes.dtype
        compute_dtype = torch.float32

        spatial_dims = monai.data.box_utils.get_spatial_dims(boxes=boxes)
        if spatial_dims == 3:
            xc, yc, zc, w, h, d = boxes.split(1, dim=-1)
            xmin = xc - ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            xmax = xc + ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymin = yc - ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = yc + ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            zmin = zc - ((d - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            zmax = zc + ((d - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            return xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xc, yc, w, h = boxes.split(1, dim=-1)
            xmin = xc - ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            xmax = xc + ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymin = yc - ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = yc + ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            return xmin, ymin, xmax, ymax

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        spatial_dims = monai.data.box_utils.get_spatial_dims(corners=corners)
        if spatial_dims == 3:
            xmin, ymin, zmin, xmax, ymax, zmax = corners[0], corners[1], corners[2], corners[3], corners[4], corners[5]
            return torch.cat(
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
            return torch.cat(
                (
                    (xmin + xmax + TO_REMOVE) / 2.0,
                    (ymin + ymax + TO_REMOVE) / 2.0,
                    xmax - xmin + TO_REMOVE,
                    ymax - ymin + TO_REMOVE,
                ),
                dim=-1,
            )
