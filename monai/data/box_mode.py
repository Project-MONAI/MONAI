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
from typing import Sequence, Tuple

import torch

# TO_REMOVE = 0 if in 'xxyy','xxyyzz' mode, the bottom-right corner is not included in the box,
#      i.e., when xmin=1, xmax=2, we have w = 1
# TO_REMOVE = 1  if in 'xxyy','xxyyzz' mode, the bottom-right corner is included in the box,
#       i.e., when xmin=1, xmax=2, we have w = 2
# Currently, only `TO_REMOVE = 0.` is supported
TO_REMOVE = 0.0  # xmax-xmin = w -TO_REMOVE.


class BoxMode:
    def __int__(self):
        # The mapping that maps spatial dimension to mode string name
        self.dim_to_str_mapping = {2: None, 3: None}

    def get_str_mode(self, spatial_dims: int) -> str:
        """
        Get the mode name for the given spatial dimension
        Args:
            spatial_dims: 2 or 3
        Returns:
            mode string name
        Example:
            boxmode.get_str_mode(spatial_dims = 2)
        """
        return self.dim_to_str_mapping[spatial_dims]

    def get_dim_from_boxes(self, boxes: torch.Tensor) -> int:
        """
        Get spatial dimension for the given boxes
        Args:
            boxes: bounding box, Nx4 or Nx6 torch tensor
        Returns:
            spatial_dims: 2 or 3
        Example:
            boxes = torch.ones(10,6)
            boxmode.get_dim_from_boxes(boxes) will return 3
        """
        if int(boxes.shape[1]) not in [4, 6]:
            raise ValueError(
                f"Currently we support only boxes with shape [N,4] or [N,6], got boxes with shape {boxes.shape}."
            )
        spatial_dims = int(boxes.shape[1] // 2)
        return spatial_dims

    def get_dim_from_corner(self, c: Sequence) -> int:
        """
        Get spatial dimension for the given box corners
        Args:
            c: corners of a box, 4-element or 6-element tuple, each element is a Nx1 torch tensor
            (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax)
        Returns:
            spatial_dims: 2 or 3
        Example:
            c = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
            boxmode.get_dim_from_corner(c) will return 2
        """
        if len(c) not in [4, 6]:
            raise ValueError(
                f"Currently we support only boxes with shape [N,4] or [N,6], got box corner tuple with length {len(c)}."
            )
        spatial_dims = int(len(c) // 2)
        return spatial_dims

    def check_corner(self, c: Sequence) -> bool:
        """
        check the validity for the given box corners
        Args:
            c: corners of a box, 4-element or 6-element tuple, each element is a Nx1 torch tensor
            (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax)
        Returns:
            bool, whether the box is valid
        Example:
            c = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
            boxmode.check_corner(c) will return True
        """
        spatial_dims = self.get_dim_from_corner(c)
        box_error = c[spatial_dims] < c[0]
        for axis in range(1, spatial_dims):
            box_error = box_error | (c[spatial_dims + axis] < c[axis])
        if box_error.sum() > 0:
            return False
        else:
            return True

    @abstractmethod
    def box_to_corner(self, boxes: torch.Tensor) -> Tuple:
        """
        Return the box corners for the given boxes
        Args:
            boxes: bounding box, Nx4 or Nx6 torch tensor
        Returns:
            corners of a box, 4-element or 6-element tuple, each element is a Nx1 torch tensor
        Example:
            boxes = torch.ones(10,6)
            boxmode.box_to_corner(boxes) will a 6-element tuple, each element is a 10x1 tensor
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def corner_to_box(self, corner: Sequence) -> torch.Tensor:
        """
        Return the boxes converted from the given box corners
        Args:
            c: corners of a box, 4-element or 6-element tuple, each element is a Nx1 torch tensor
            (xmin, ymin, xmax, ymax) or (xmin, ymin, zmin, xmax, ymax, zmax)
        Returns:
            boxes: bounding box, Nx4 or Nx6 torch tensor
        Example:
            c = (torch.ones(10,1), torch.ones(10,1), torch.ones(10,1), torch.ones(10,1))
            boxmode.corner_to_box(c) will return a 10x4 tensor
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class CornerCornerMode_TypeA(BoxMode):
    """
    Also represented as "xyxy" or "xyzxyz"
    [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax]
    """

    def __int__(self):
        self.dim_to_str_mapping = {2: "xyxy", 3: "xyzxyx"}

    def box_to_corner(self, boxes: torch.Tensor) -> Tuple:
        spatial_dims = self.get_dim_from_boxes(boxes)
        if spatial_dims == 3:
            xmin, ymin, zmin, xmax, ymax, zmax = boxes.split(1, dim=-1)
            corner = xmin, ymin, zmin, xmax, ymax, zmax
        if spatial_dims == 2:
            xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)
            corner = xmin, ymin, xmax, ymax
        if self.check_corner(corner):
            return corner
        else:
            raise ValueError("Given boxes has invalid values. The box size must be non-negative.")

    def corner_to_box(self, c: Sequence) -> torch.Tensor:
        spatial_dims = self.get_dim_from_corner(c)
        if spatial_dims == 3:
            return torch.cat((c[0], c[1], c[2], c[3], c[4], c[5]), dim=-1)
        if spatial_dims == 2:
            return torch.cat((c[0], c[1], c[2], c[3]), dim=-1)


class CornerCornerMode_TypeB(BoxMode):
    """
    Also represented as "xxyy" or "xxyyzz"
    [xmin, xmax, ymin, ymax] or [xmin, xmax, ymin, ymax, zmin, zmax]
    """

    def __int__(self):
        self.dim_to_str_mapping = {2: "xxyy", 3: "xxyyzz"}

    def box_to_corner(self, boxes: torch.Tensor) -> Tuple:
        spatial_dims = self.get_dim_from_boxes(boxes)
        if spatial_dims == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = boxes.split(1, dim=-1)
            corner = xmin, ymin, zmin, xmax, ymax, zmax
        if spatial_dims == 2:
            xmin, xmax, ymin, ymax = boxes.split(1, dim=-1)
            corner = xmin, ymin, xmax, ymax
        if self.check_corner(corner):
            return corner
        else:
            raise ValueError("Given boxes has invalid values. The box size must be non-negative.")

    def corner_to_box(self, c: Sequence) -> torch.Tensor:
        spatial_dims = self.get_dim_from_corner(c)
        if spatial_dims == 3:
            return torch.cat((c[0], c[3], c[1], c[4], c[2], c[5]), dim=-1)
        if spatial_dims == 2:
            return torch.cat((c[0], c[2], c[1], c[3]), dim=-1)


class CornerCornerMode_TypeC(BoxMode):
    """
    Also represented as "xyxy" or "xyxyzz"
    [xmin, ymin, xmax, ymax] or [xmin, ymin, xmax, ymax, zmin, zmax]
    """

    def __int__(self):
        self.dim_to_str_mapping = {2: "xyxy", 3: "xyxyzz"}

    def box_to_corner(self, boxes: torch.Tensor) -> Tuple:
        spatial_dims = self.get_dim_from_boxes(boxes)
        if spatial_dims == 3:
            xmin, ymin, xmax, ymax, zmin, zmax = boxes.split(1, dim=-1)
            corner = xmin, ymin, zmin, xmax, ymax, zmax
        if spatial_dims == 2:
            xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)
            corner = xmin, ymin, xmax, ymax
        if self.check_corner(corner):
            return corner
        else:
            raise ValueError("Given boxes has invalid values. The box size must be non-negative.")

    def corner_to_box(self, c: Sequence) -> torch.Tensor:
        spatial_dims = self.get_dim_from_corner(c)
        if spatial_dims == 3:
            return torch.cat((c[0], c[1], c[3], c[4], c[2], c[5]), dim=-1)
        if spatial_dims == 2:
            return torch.cat((c[0], c[1], c[2], c[3]), dim=-1)


class CornerSizeMode(BoxMode):
    """
    Also represented as "xywh" or "xyzwhd"
    [xmin, ymin, xsize, ysize] or [xmin, ymin, zmin, xsize, ysize, zsize]
    """

    def __int__(self):
        self.dim_to_str_mapping = {2: "xywh", 3: "xyzwhd"}

    def box_to_corner(self, boxes: torch.Tensor) -> Tuple:
        # convert to float32 when computing torch.clamp, which does not support float16
        box_dtype = boxes.dtype
        compute_dtype = torch.float32

        spatial_dims = self.get_dim_from_boxes(boxes)
        if spatial_dims == 3:
            xmin, ymin, zmin, w, h, d = boxes.split(1, dim=-1)
            xmax = xmin + (w - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = ymin + (h - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            zmax = zmin + (d - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            corner = xmin, ymin, zmin, xmax, ymax, zmax
        if spatial_dims == 2:
            xmin, ymin, w, h = boxes.split(1, dim=-1)
            xmax = xmin + (w - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = ymin + (h - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            corner = xmin, ymin, xmax, ymax
        if self.check_corner(corner):
            return corner
        else:
            raise ValueError("Given boxes has invalid values. The box size must be non-negative.")

    def corner_to_box(self, c: Sequence) -> torch.Tensor:
        spatial_dims = self.get_dim_from_corner(c)
        if spatial_dims == 3:
            xmin, ymin, zmin, xmax, ymax, zmax = c[0], c[1], c[2], c[3], c[4], c[5]
            return torch.cat(
                (xmin, ymin, zmin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE, zmax - zmin + TO_REMOVE), dim=-1
            )
        if spatial_dims == 2:
            xmin, ymin, xmax, ymax = c[0], c[1], c[2], c[3]
            return torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)


class CenterSizeMode(BoxMode):
    """
    Also represented as "ccwh" or "cccwhd"
    [xmin, ymin, xsize, ysize] or [xmin, ymin, zmin, xsize, ysize, zsize]
    """

    def __int__(self):
        self.dim_to_str_mapping = {2: "ccwh", 3: "cccwhd"}

    def box_to_corner(self, boxes: torch.Tensor) -> Tuple:
        # convert to float32 when computing torch.clamp, which does not support float16
        box_dtype = boxes.dtype
        compute_dtype = torch.float32

        spatial_dims = self.get_dim_from_boxes(boxes)
        if spatial_dims == 3:
            xc, yc, zc, w, h, d = boxes.split(1, dim=-1)
            xmin = xc - ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            xmax = xc + ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymin = yc - ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = yc + ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            zmin = zc - ((d - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            zmax = zc + ((d - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            corner = xmin, ymin, zmin, xmax, ymax, zmax
        if spatial_dims == 2:
            xc, yc, w, h = boxes.split(1, dim=-1)
            xmin = xc - ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            xmax = xc + ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymin = yc - ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            ymax = yc + ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype)
            corner = xmin, ymin, xmax, ymax
        if self.check_corner(corner):
            return corner
        else:
            raise ValueError("Given boxes has invalid values. The box size must be non-negative.")

    def corner_to_box(self, c: Sequence) -> torch.Tensor:
        spatial_dims = int(len(c) // 2)
        if spatial_dims == 3:
            xmin, ymin, zmin, xmax, ymax, zmax = c[0], c[1], c[2], c[3], c[4], c[5]
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
        if spatial_dims == 2:
            xmin, ymin, xmax, ymax = c[0], c[1], c[2], c[3]
            return torch.cat(
                (
                    (xmin + xmax + TO_REMOVE) / 2.0,
                    (ymin + ymax + TO_REMOVE) / 2.0,
                    xmax - xmin + TO_REMOVE,
                    ymax - ymin + TO_REMOVE,
                ),
                dim=-1,
            )
