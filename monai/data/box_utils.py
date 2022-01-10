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

from monai.utils.module import look_up_option

SUPPORT_MODE = ["xxyy", "xxyyzz", "xyxy", "xyzxyz", "xywh", "xyzwhd"]
STANDARD_MODE = ["xxyy", "xxyyzz"]  # [2d_mode, 3d_mode]

# TO_REMOVE = 0 if in 'xxyy','xxyyzz' mode, the bottom-right corner is not included in the box,
#      i.e., when x_min=1, x_max=2, we have w = 1
# TO_REMOVE = 1  if in 'xxyy','xxyyzz' mode, the bottom-right corner is included in the box,
#       i.e., when x_min=1, x_max=2, we have w = 2
TO_REMOVE = 0  # x_max-x_min = w -TO_REMOVE

"""
The following variables share the same definition across the functions in this file.
Args:
    bbox: Nx4 or Nx6 torch tensor
    mode: choose from SUPPORT_MODE. If mode is not given, these funcs will assume mode is STANDARD_MODE
    image_size: Length of 2 or 3. Data format is list, or np.ndarray, or tensor of int
"""


def convert_to_list(in_sequence: Union[Sequence, torch.Tensor, np.ndarray]) -> list:
    """
    convert a torch.Tensor, or np array input to list
    Args:
        in_sequence:
    Returns: in_sequence_list

    """
    in_sequence_list = deepcopy(in_sequence)
    if torch.is_tensor(in_sequence):
        in_sequence_list.cpu().detach().numpy().tolist()
    elif isinstance(in_sequence, np.ndarray):
        in_sequence_list.tolist()
    elif not isinstance(in_sequence, list):
        in_sequence_list = list(in_sequence_list)
    return in_sequence_list


def get_dimension(
    bbox: torch.Tensor = None, image_size: Union[Sequence[int], torch.Tensor, np.ndarray] = None, mode: str = None
) -> int:
    """
    Get spatial dimension for the giving setting.
    Missing input is allowed. But at least one of the input value should be given.
    Returns: spatial_dimension
    """
    spatial_dims = set()
    if image_size is not None:
        spatial_dims.add(len(image_size))
    if mode is not None:
        spatial_dims.add(len(mode) / 2)
    if bbox is not None:
        spatial_dims.add(int(bbox.shape[1] / 2))
    spatial_dims = list(spatial_dims)
    if len(spatial_dims) == 0:
        raise ValueError("At least one of bbox, image_size, and mode needs to be non-empty.")
    elif len(spatial_dims) == 1:
        spatial_dims = int(spatial_dims[0])
        spatial_dims = look_up_option(spatial_dims,supported=[2,3])
        return int(spatial_dims)
    else:
        raise ValueError("The dimension of bbox, image_size, mode should match with each other.")


def get_standard_mode(spatial_dims: int) -> str:
    """
    Get the mode name for the given spatial dimension
    Args:
        spatial_dims: 2 or 3

    Returns: mode

    """
    if spatial_dims == 2:
        return STANDARD_MODE[0]
    elif spatial_dims == 3:
        return STANDARD_MODE[1]
    else:
        ValueError(f"Images should have 2 or 3 dimensions, got {spatial_dims}")


def point_interp(
    point1: Union[Sequence, torch.Tensor, np.ndarray], zoom: Union[Sequence[float], float]
) -> Union[Sequence, torch.Tensor, np.ndarray]:
    """
    Convert point position from one pixel/voxel size to another pixel/voxel size
    Args:
        point1: point coordinate on an image with pixel/voxel size of pix_size1
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
    Returns:
        point2: point coordinate on an image with pixel/voxel size of pix_size2
    """
    # make sure the spatial dimensions of the inputs match with each other
    spatial_dims = len(point1)
    spatial_dims = look_up_option(spatial_dims,supported=[2,3])

    # compute new point
    point2 = deepcopy(point1)
    _zoom = monai.utils.misc.ensure_tuple_rep(zoom, spatial_dims)
    for axis in range(0, spatial_dims):
        point2[axis] = point1[axis] * _zoom[axis]
    return point2


def box_interp(bbox1: torch.Tensor, zoom: Union[Sequence[float], float], mode1: str = None) -> torch.Tensor:
    """
    Interpolate bbox
    Args:
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.

    Returns:
        bbox2: returned bbox has the same mode as bbox1
    """
    if mode1 is None:
        mode1 = get_standard_mode(int(bbox1.shape[1] / 2))
    mode1 = look_up_option(mode1,supported=SUPPORT_MODE)
    spatial_dims = get_dimension(bbox=bbox1, mode=mode1)

    mode_standard = get_standard_mode(spatial_dims)
    bbox1_standard = box_convert_mode(bbox1=bbox1, mode1=mode1, mode2=mode_standard)

    corner_lt = point_utils.point_interp(bbox1_standard[:, ::2], zoom)
    corner_rb = point_utils.point_interp(bbox1_standard[:, 1::2], zoom)

    bbox2_standard_interp = deepcopy(bbox2_standard)
    bbox2_standard_interp[:, ::2] = corner_lt
    bbox2_standard_interp[:, 1::2] = corner_rb

    return box_convert_mode(bbox1=bbox2_standard_interp, mode1=mode_standard, mode2=mode1)


def split_into_corners(bbox: torch.Tensor, mode: str):
    """
    This internal function outputs the corner coordinates of the bbox

    Returns:
        if 2D image, outputs (xmin, xmax, ymin, ymax)
        if 3D images, outputs (xmin, xmax, ymin, ymax, zmin, zmax)
        xmin for example, is a Nx1 tensor

    """
    mode = look_up_option(mode,supported=SUPPORT_MODE)
    if mode in STANDARD_MODE:
        return bbox.split(1, dim=-1)
    elif mode == "xyzxyz":
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.split(1, dim=-1)
        return (
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        )
    elif mode == "xyxy":
        xmin, ymin, xmax, ymax = bbox.split(1, dim=-1)
        return (xmin, xmax, ymin, ymax)
    elif mode == "xyzwhd":
        xmin, ymin, zmin, w, h, d = bbox.split(1, dim=-1)
        return (
            xmin,
            xmin + (w - TO_REMOVE).clamp(min=0),
            ymin,
            ymin + (h - TO_REMOVE).clamp(min=0),
            zmin,
            zmin + (d - TO_REMOVE).clamp(min=0),
        )
    elif mode == "xywh":
        xmin, ymin, w, h = bbox.split(1, dim=-1)
        return (xmin, xmin + (w - TO_REMOVE).clamp(min=0), ymin, ymin + (h - TO_REMOVE).clamp(min=0))
    else:
        raise RuntimeError("Should not be here")


def box_convert_mode(bbox1: torch.Tensor, mode1: str, mode2: str) -> torch.Tensor:
    """
    This function converts the bbox1 in mode 1 to the mode2
    """
    # 1. check whether the bbox and the new mode is valid
    if mode1 is None:
        mode1 = get_standard_mode(int(bbox1.shape[1] / 2))
    if mode2 is None:
        mode2 = get_standard_mode(int(bbox1.shape[1] / 2))
    mode1 = look_up_option(mode1,supported=SUPPORT_MODE)
    mode2 = look_up_option(mode2,supported=SUPPORT_MODE)

    spatial_dims = get_dimension(bbox=bbox1, mode=mode1)
    if len(mode1) != len(mode2):
        raise ValueError("The dimension of the new mode should have the same spatial dimension as the old mode.")

    # 2. if mode not changed, return original boxlist
    if mode1 == mode2:
        return deepcopy(bbox1)

    # 3. convert mode for bbox
    if mode2 in STANDARD_MODE:
        corners = split_into_corners(deepcopy(bbox1), mode1)
        return torch.cat(corners, dim=-1)

    if spatial_dims == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = split_into_corners(deepcopy(bbox1), mode1)
        if mode2 == "xyzxyz":
            bbox2 = torch.cat((xmin, ymin, zmin, xmax, ymax, zmax), dim=-1)
        elif mode2 == "xyzwhd":
            bbox2 = torch.cat(
                (xmin, ymin, zmin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE, zmax - zmin + TO_REMOVE), dim=-1
            )
        else:
            raise ValueError("We support only bbox mode in " + str(SUPPORT_MODE) + f", got {mode2}")
    elif spatial_dims == 2:
        xmin, xmax, ymin, ymax = split_into_corners(deepcopy(bbox1), mode1)
        if mode2 == "xyxy":
            bbox2 = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
        elif mode2 == "xywh":
            bbox2 = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)
        else:
            raise ValueError("We support only bbox mode in " + str(SUPPORT_MODE) + f", got {mode2}")
    else:
        raise ValueError(f"Images should have 2 or 3 dimensions, got {spatial_dims}")

    return bbox2


def box_convert_standard_mode(bbox: torch.Tensor, mode: str) -> torch.Tensor:
    """
    This function convert the bbox in mode 1 to 'xyxy' or 'xyzxyz'
    """
    mode = look_up_option(mode,supported=SUPPORT_MODE)
    spatial_dims = get_dimension(bbox=bbox, mode=mode)
    mode_standard = get_standard_mode(spatial_dims)
    return box_convert_mode(bbox1=bbox, mode1=mode, mode2=mode_standard)


def box_area(bbox: torch.Tensor, mode: str = None) -> torch.tensor:
    """
    This function computes the area of each box
    Returns:
        area: 1-D tensor
    """

    if mode is None:
        mode = get_standard_mode(int(bbox.shape[1] / 2))
    mode = look_up_option(mode,supported=STANDARD_MODE)
    spatial_dims = get_dimension(bbox=bbox, mode=mode)

    area = bbox[:, 1] - bbox[:, 0] + TO_REMOVE
    for axis in range(1, spatial_dims):
        area = area * (bbox[:, 2 * axis + 1] - bbox[:, 2 * axis] + TO_REMOVE)

    return area


def box_clip_to_image(
    bbox: torch.Tensor,
    image_size: Union[Sequence[int], torch.Tensor, np.ndarray],
    mode: str = None,
    remove_empty: bool = True,
) -> dict:
    """
    This function makes sure the bounding boxes are within the image.
    Args:
        remove_empty: whether to remove the boxes that are actually empty
    Returns:
        updated box
    """
    if mode is None:
        mode = get_standard_mode(int(bbox.shape[1] / 2))
    mode = look_up_option(mode,supported=STANDARD_MODE)
    spatial_dims = get_dimension(bbox=bbox, image_size=image_size, mode=mode)
    new_bbox = deepcopy(bbox)
    if bbox.shape[0] == 0:
        return deepcopy(bbox)

    # 1. convert to standard mode
    mode_standard = get_standard_mode(spatial_dims)
    new_bbox = box_convert_mode(bbox1=new_bbox, mode1=mode, mode2=mode_standard)

    # 2. makes sure the bounding boxes are within the image
    for axis in range(0, spatial_dims):
        new_bbox[:, 2 * axis].clamp_(min=0, max=image_size[axis] - TO_REMOVE)
        new_bbox[:, 2 * axis + 1].clamp_(min=0, max=image_size[axis] - TO_REMOVE)

    # 3. remove the boxes that are actually empty
    if remove_empty:
        keep = (new_bbox[:, 1] > new_bbox[:, 0]) & (new_bbox[:, 3] > new_bbox[:, 2])
        if spatial_dims == 3:
            keep = keep & (new_bbox[:, 5] > new_bbox[:, 4])
        new_bbox = new_bbox[keep]

    # 4. return updated boxlist
    new_bbox = box_convert_mode(bbox1=new_bbox, mode1=mode_standard, mode2=mode)

    return new_bbox


def box_iou(bbox1: torch.Tensor, bbox2: torch.Tensor, mode1: str = None, mode2: str = None, gpubool: bool = True):
    """
    Compute the intersection over union of two set of boxes. This function is not differentialable.

    IMPORTANT: Please run box_clip_to_image(bbox, image_size, mode, remove_empty=True) before computing IoU

    Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    with slight modifications.

    Arguments:
      bbox1: Nx4 or Nx6, make sure they are non-empty
      bbox2: Mx4 or Mx6, make sure they are non-empty
      gpubool: whether to send the final IoU results to GPU

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """

    if mode1 is None:
        mode1 = get_standard_mode(int(bbox1.shape[1] / 2))
    if mode2 is None:
        mode2 = get_standard_mode(int(bbox2.shape[1] / 2))
    mode1 = look_up_option(mode1,supported=STANDARD_MODE)
    mode2 = look_up_option(mode2,supported=STANDARD_MODE)
    spatial_dims = get_dimension(bbox=bbox1, mode=mode1)

    # we do computation on cpu
    device = bbox1.device

    # compute area for the bbox
    area1 = box_area(bbox=bbox1, mode=mode1).cpu()  # Nx1
    area2 = box_area(bbox=bbox2, mode=mode2).cpu()  # Mx1

    # get the left top and right bottom points for the NxM combinations
    lt = torch.max(bbox1[:, None, ::2], bbox2[:, ::2])  # [N,M,spatial_dims] left top
    rb = torch.min(bbox1_corner[:, None, 1::2], bbox2_corner[:, 1::2])  # [N,M,spatial_dims] right bottom
    # compute size for the intersection region for the NxM combinations
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,spatial_dims]
    inter = wh[:, :, 0]  # [N,M]
    for axis in range(1, spatial_dims):
        inter = inter * wh[:, :, axis]

    # compute IoU
    iou = inter / (area1[:, None] + area2 - inter + torch.finfo(torch.float32).eps)  # [N,M,spatial_dims]

    if gpubool:
        iou = iou.to(device)  # [N,M,spatial_dims]

    return iou
