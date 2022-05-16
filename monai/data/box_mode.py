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

from abc import abstractmethod
from typing import Dict, Sequence, Tuple

import torch

import monai
from monai.utils.enums import BoundingBoxMode

# TO_REMOVE = 0.0 if the bottom-right corner pixel/voxel is not included in the box,
#      i.e., when xmin=1., xmax=2., we have w = 1.
# TO_REMOVE = 1.0  if the bottom-right corner pixel/voxel is included in the box,
#       i.e., when xmin=1., xmax=2., we have w = 2.
# Currently, only `TO_REMOVE = 0.0` is supported
TO_REMOVE = 0.0  # xmax-xmin = w -TO_REMOVE.


class BoxMode:
    """
    An abstract class of a ``BoxMode``.
    A BoxMode is callable that converts box mode of boxes.
    It always creates a copy and will not modify boxes in place,
    the implementation should be aware of:
        #. remember to define ``name`` which is a dictionary that maps spatial_dims to box mode string
    """

    name: Dict[int, str] = {}

    @classmethod
    def get_name(cls, spatial_dims: int) -> str:
        """
        Get the mode name for the given spatial dimension
        Args:
            spatial_dims: 2 or 3
        Returns:
            mode string name
        Example:
            BoxMode.get_name(spatial_dims = 2)
        """
        return cls.name[spatial_dims].value

    @abstractmethod
    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        """
        Return the box corners for the given boxes
        Args:
            boxes: bounding box, Nx4 or Nx6 torch tensor
        Returns:
            corners of a box, 4-element or 6-element tuple, each element is a Nx1 torch tensor
        Example:
            boxmode = BoxMode()
            boxes = torch.ones(10,6)
            boxmode.boxes_to_corners(boxes) will a 6-element tuple, each element is a 10x1 tensor
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        """
        Return the boxes converted from the given box corners
        Args:
            corners: corners of a box, 4-element or 6-element tuple, each element is a Nx1 torch tensor
            (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax)
        Returns:
            boxes: bounding box, Nx4 or Nx6 torch tensor
        Example:
            boxmode = BoxMode()
            corners = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
            boxmode.corners_to_boxes(corners) will return a 10x4 tensor
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class CornerCornerModeTypeA(BoxMode):
    """
    Also represented as "xyxy" or "xyzxyz"
    [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax]
    """

    name = {2: BoundingBoxMode.XYXY, 3: BoundingBoxMode.XYZXYZ}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        spatial_dims = monai.data.box_utils.get_dimension(boxes=boxes)
        if spatial_dims == 3:
            xmin, ymin, zmin, xmax, ymax, zmax = boxes.split(1, dim=-1)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)
            corners = xmin, ymin, xmax, ymax
        return corners

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        spatial_dims = monai.data.box_utils.get_dimension(corners=corners)
        if spatial_dims == 3:
            return torch.cat((corners[0], corners[1], corners[2], corners[3], corners[4], corners[5]), dim=-1)
        elif spatial_dims == 2:
            return torch.cat((corners[0], corners[1], corners[2], corners[3]), dim=-1)


class CornerCornerModeTypeB(BoxMode):
    """
    Also represented as "xxyy" or "xxyyzz"
    [xmin, xmax, ymin, ymax] or [xmin, xmax, ymin, ymax, zmin, zmax]
    """

    name = {2: BoundingBoxMode.XXYY, 3: BoundingBoxMode.XXYYZZ}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        spatial_dims = monai.data.box_utils.get_dimension(boxes=boxes)
        if spatial_dims == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = boxes.split(1, dim=-1)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xmin, xmax, ymin, ymax = boxes.split(1, dim=-1)
            corners = xmin, ymin, xmax, ymax
        return corners

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        spatial_dims = monai.data.box_utils.get_dimension(corners=corners)
        if spatial_dims == 3:
            return torch.cat((corners[0], corners[3], corners[1], corners[4], corners[2], corners[5]), dim=-1)
        elif spatial_dims == 2:
            return torch.cat((corners[0], corners[2], corners[1], corners[3]), dim=-1)


class CornerCornerModeTypeC(BoxMode):
    """
    Also represented as "xyxy" or "xyxyzz"
    [xmin, ymin, xmax, ymax] or [xmin, ymin, xmax, ymax, zmin, zmax]
    """

    name = {2: BoundingBoxMode.XYXY, 3: BoundingBoxMode.XYXYZZ}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        spatial_dims = monai.data.box_utils.get_dimension(boxes=boxes)
        if spatial_dims == 3:
            xmin, ymin, xmax, ymax, zmin, zmax = boxes.split(1, dim=-1)
            corners = xmin, ymin, zmin, xmax, ymax, zmax
        elif spatial_dims == 2:
            xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)
            corners = xmin, ymin, xmax, ymax
        return corners

    def corners_to_boxes(self, corners: Sequence) -> torch.Tensor:
        spatial_dims = monai.data.box_utils.get_dimension(corners=corners)
        if spatial_dims == 3:
            return torch.cat((corners[0], corners[1], corners[3], corners[4], corners[2], corners[5]), dim=-1)
        elif spatial_dims == 2:
            return torch.cat((corners[0], corners[1], corners[2], corners[3]), dim=-1)


class CornerSizeMode(BoxMode):
    """
    Also represented as "xywh" or "xyzwhd"
    [xmin, ymin, xsize, ysize] or [xmin, ymin, zmin, xsize, ysize, zsize]
    """

    name = {2: BoundingBoxMode.XYWH, 3: BoundingBoxMode.XYZWHD}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        # convert to float32 when computing torch.clamp, which does not support float16
        box_dtype = boxes.dtype
        compute_dtype = torch.float32

        spatial_dims = monai.data.box_utils.get_dimension(boxes=boxes)
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
        spatial_dims = monai.data.box_utils.get_dimension(corners=corners)
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
    Also represented as "ccwh" or "cccwhd"
    [xmin, ymin, xsize, ysize] or [xmin, ymin, zmin, xsize, ysize, zsize]
    """

    name = {2: BoundingBoxMode.CCWH, 3: BoundingBoxMode.CCCWHD}

    def boxes_to_corners(self, boxes: torch.Tensor) -> Tuple:
        # convert to float32 when computing torch.clamp, which does not support float16
        box_dtype = boxes.dtype
        compute_dtype = torch.float32

        spatial_dims = monai.data.box_utils.get_dimension(boxes=boxes)
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
        spatial_dims = monai.data.box_utils.get_dimension(corners=corners)
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
