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
from typing import List, Sequence, Union

import numpy as np
import torch

import monai
from monai.utils.module import look_up_option
from monai.config.type_definitions import NdarrayOrTensor

CORNER_CORNER_MODE = ["xyxy", "xyzxyz"]  # [2d_mode, 3d_mode]
XXYYZZ_MODE = ["xxyy", "xxyyzz"]  # [2d_mode, 3d_mode]
CORNER_SIZE_MODE = ["xywh", "xyzwhd"] # [2d_mode, 3d_mode]
CENTER_SIZE_MODE = ["ccwh", "cccwhd"] # [2d_mode, 3d_mode]

STANDARD_MODE = CORNER_CORNER_MODE  # [2d_mode, 3d_mode]
SUPPORT_MODE = CORNER_CORNER_MODE + XXYYZZ_MODE + CORNER_SIZE_MODE + CENTER_SIZE_MODE

# TO_REMOVE = 0 if in 'xxyy','xxyyzz' mode, the bottom-right corner is not included in the box,
#      i.e., when x_min=1, x_max=2, we have w = 1
# TO_REMOVE = 1  if in 'xxyy','xxyyzz' mode, the bottom-right corner is included in the box,
#       i.e., when x_min=1, x_max=2, we have w = 2
# Currently only TO_REMOVE = 0 has been tested. Please use TO_REMOVE = 0
TO_REMOVE = 0  # x_max-x_min = w -TO_REMOVE.


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
        in_sequence: Sequence or torch.Tensor or np.ndarray
    Returns: in_sequence_list

    """
    in_sequence_list = deepcopy(in_sequence)
    if torch.is_tensor(in_sequence):
        in_sequence_list.detach().cpu().numpy().tolist()
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
    Returns: spatial_dimension, 2 or 3
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
        spatial_dims = look_up_option(spatial_dims, supported=[2, 3])
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
        raise ValueError(f"Images should have 2 or 3 dimensions, got {spatial_dims}")


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
    spatial_dims = look_up_option(spatial_dims, supported=[2, 3])

    # compute new point
    point2 = deepcopy(point1)
    _zoom = monai.utils.misc.ensure_tuple_rep(zoom, spatial_dims)
    for axis in range(0, spatial_dims):
        point2[axis] = point1[axis] * _zoom[axis]
    return point2


def box_interp(bbox: torch.Tensor, zoom: Union[Sequence[float], float], mode: str = None) -> torch.Tensor:
    """
    Interpolate bbox
    Args:
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.

    Returns:
        bbox2: returned bbox has the same mode as bbox
    """
    if mode is None:
        mode = get_standard_mode(int(bbox.shape[1] / 2))
    mode = look_up_option(mode, supported=SUPPORT_MODE)
    spatial_dims = get_dimension(bbox=bbox, mode=mode)

    mode_standard = get_standard_mode(spatial_dims)
    bbox_standard = box_convert_mode(bbox1=bbox, mode1=mode, mode2=mode_standard)

    corner_lt = point_interp(bbox_standard[:, :spatial_dims], zoom)
    corner_rb = point_interp(bbox_standard[:, spatial_dims:], zoom)

    bbox_standard_interp = deepcopy(bbox_standard)
    bbox_standard_interp[:, :spatial_dims] = corner_lt
    bbox_standard_interp[:, spatial_dims:] = corner_rb

    return box_convert_mode(bbox1=bbox_standard_interp, mode1=mode_standard, mode2=mode)


def split_into_corners(bbox: torch.Tensor, mode: str = None):
    """
    This internal function outputs the corner coordinates of the bbox

    Returns:
        if 2D image, outputs (xmin, xmax, ymin, ymax)
        if 3D images, outputs (xmin, xmax, ymin, ymax, zmin, zmax)
        xmin for example, is a Nx1 tensor

    """
    if mode is None:
        mode = get_standard_mode(int(bbox.shape[1] / 2))
    mode = look_up_option(mode, supported=SUPPORT_MODE)
    if mode in ["xxyy","xxyyzz"]:
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
    elif mode == "cccwhd":
        xc, yc, zc, w, h, d = bbox.split(1, dim=-1)
        return (
            xc - ((w - TO_REMOVE)/2.0).clamp(min=0),
            xc + ((w - TO_REMOVE)/2.0).clamp(min=0),
            yc - ((h - TO_REMOVE)/2.0).clamp(min=0),
            yc + ((h - TO_REMOVE)/2.0).clamp(min=0),
            zc - ((d - TO_REMOVE)/2.0).clamp(min=0),
            zc + ((d - TO_REMOVE)/2.0).clamp(min=0)
        )
    elif mode == "ccwh":
        xc, yc, w, h = bbox.split(1, dim=-1)
        return (
            xc - ((w - TO_REMOVE)/2.0).clamp(min=0),
            xc + ((w - TO_REMOVE)/2.0).clamp(min=0),
            yc - ((h - TO_REMOVE)/2.0).clamp(min=0),
            yc + ((h - TO_REMOVE)/2.0).clamp(min=0)
        )
    else:
        raise RuntimeError("Should not be here")


def box_convert_mode(bbox1: torch.Tensor, mode1: str = None, mode2: str = None) -> torch.Tensor:
    """
    This function converts the bbox1 in mode 1 to the mode2
    """
    # 1. check whether the bbox and the new mode is valid
    if mode1 is None:
        mode1 = get_standard_mode(int(bbox1.shape[1] / 2))
    if mode2 is None:
        mode2 = get_standard_mode(int(bbox1.shape[1] / 2))
    mode1 = look_up_option(mode1, supported=SUPPORT_MODE)
    mode2 = look_up_option(mode2, supported=SUPPORT_MODE)

    spatial_dims = get_dimension(bbox=bbox1, mode=mode1)
    if len(mode1) != len(mode2):
        raise ValueError("The dimension of the new mode should have the same spatial dimension as the old mode.")

    # 2. if mode not changed, return original boxlist
    if mode1 == mode2:
        return deepcopy(bbox1)

    # 3. convert mode for bbox
    if mode2 in ["xxyy","xxyyzz"]:
        corners = split_into_corners(bbox1.clone(), mode1)
        return torch.cat(corners, dim=-1)

    if spatial_dims == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = split_into_corners(bbox1.clone(), mode1)
        if mode2 == "xyzxyz":
            bbox2 = torch.cat((xmin, ymin, zmin, xmax, ymax, zmax), dim=-1)
        elif mode2 == "xyzwhd":
            bbox2 = torch.cat(
                (xmin, ymin, zmin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE, zmax - zmin + TO_REMOVE), dim=-1
            )
        elif mode2 == "cccwhd":
            bbox2 = torch.cat(
                (
                    (xmin+xmax+ TO_REMOVE)/2, (ymin+ymax+ TO_REMOVE)/2, (zmin+zmax+ TO_REMOVE)/2,
                    xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE, zmax - zmin + TO_REMOVE
                ), dim=-1
            )
        else:
            raise ValueError("We support only bbox mode in " + str(SUPPORT_MODE) + f", got {mode2}")
    elif spatial_dims == 2:
        xmin, xmax, ymin, ymax = split_into_corners(bbox1.clone(), mode1)
        if mode2 == "xyxy":
            bbox2 = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
        elif mode2 == "xywh":
            bbox2 = torch.cat((xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1)
        elif mode2 == "ccwh":
            bbox2 = torch.cat(
                (
                    (xmin+xmax+ TO_REMOVE)/2, (ymin+ymax+ TO_REMOVE)/2, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE
                ), dim=-1
            )
        else:
            raise ValueError("We support only bbox mode in " + str(SUPPORT_MODE) + f", got {mode2}")
    else:
        raise ValueError(f"Images should have 2 or 3 dimensions, got {spatial_dims}")

    return bbox2


def box_convert_standard_mode(bbox: torch.Tensor, mode: str) -> torch.Tensor:
    """
    This function convert the bbox in mode 1 to 'xyxy' or 'xyzxyz'
    """
    mode = look_up_option(mode, supported=SUPPORT_MODE)
    spatial_dims = get_dimension(bbox=bbox, mode=mode)
    mode_standard = get_standard_mode(spatial_dims)
    return box_convert_mode(bbox1=bbox, mode1=mode, mode2=mode_standard)




def box_affine(bbox: torch.Tensor, affine: torch.Tensor, mode: str) -> torch.Tensor:
    """
    This function applys affine matrixs to the bbox
    Args:
        affine: affine matric to be applied to the box coordinate, (spatial_dims+1)x(spatial_dims+1)
    """
    if mode is None:
        mode = get_standard_mode(int(bbox.shape[1] / 2))
    mode = look_up_option(mode, supported=SUPPORT_MODE)
    spatial_dims = get_dimension(bbox=bbox, mode=mode)


    if  mode in ["xxyy","xxyyzz","xyxy","xyzxyz"]:
        if mode in ["xxyy","xxyyzz"]:
            lt = torch.cat([bbox[:, ::2],torch.ones(bbox.shape[0],1)],dim=1).transpose(0, 1)
            rb = torch.cat([bbox[:, 1::2],torch.ones(bbox.shape[0],1)],dim=1).transpose(0, 1)
        if mode in ["xyxy","xyzxyz"]:
            lt = torch.cat([bbox[:, :spatial_dims],torch.ones(bbox.shape[0],1)],dim=1).transpose(0, 1)
            rb = torch.cat([bbox[:, spatial_dims:],torch.ones(bbox.shape[0],1)],dim=1).transpose(0, 1)


        lt_new = torch.matmul(affine,lt)
        rb_new = torch.matmul( affine,rb)

        lt = lt_new[:spatial_dims,:].transpose(0, 1)
        rb = rb_new[:spatial_dims,:].transpose(0, 1)

        lt_new,_ = torch.min(torch.stack([lt,rb],dim=2),dim=2)
        rb_new,_ = torch.max(torch.stack([lt,rb],dim=2),dim=2)

        return box_convert_mode(torch.cat([lt_new,rb_new],dim=1), mode1=STANDARD_MODE[spatial_dims-2], mode2=mode)


    elif mode in ["ccwh", "cccwhd"]:
        lt = torch.cat([bbox[:, :spatial_dims],torch.ones(bbox.shape[0],1)],dim=1).transpose(0, 1)
        wh = bbox[:, spatial_dims:].transpose(0, 1)


        lt_new = torch.matmul( affine,lt)
        wh_new =torch.matmul( affine[:spatial_dims,:spatial_dims],wh)

        lt_new = lt_new[:spatial_dims,:].transpose(0, 1)
        wh_new = wh_new.transpose(0, 1).absolute()


        return torch.cat([lt_new,wh_new],dim=1)

    else:
        raise RuntimeError("Should not be here")




## --------------------------------------------------------------------------- ##
# In the following script, all the box mode is assumed to be STANDARD_MODE

def box_clip_to_patch(
    bbox: torch.Tensor,
    patch_box: Union[Sequence[int], torch.Tensor, np.ndarray],
    remove_empty: bool = True,
) -> dict:
    """
    This function makes sure the bounding boxes are within the image.
    Args:
        remove_empty: whether to remove the boxes that are actually empty
    Returns:
        updated box
    """
    spatial_dims = get_dimension(bbox=bbox)
    new_bbox = deepcopy(bbox)

    if bbox.shape[0] == 0:
        return deepcopy(bbox), []
    # makes sure the bounding boxes are within the image
    for axis in range(0, spatial_dims):
        new_bbox[:, axis].clamp_(min=patch_box[axis], max=patch_box[axis+spatial_dims] - TO_REMOVE)
        new_bbox[:, axis + spatial_dims].clamp_(min=patch_box[axis], max=patch_box[axis+spatial_dims] - TO_REMOVE)
        new_bbox[:, axis] -= patch_box[axis]
        new_bbox[:, axis + spatial_dims] -= patch_box[axis]

    # remove the boxes that are actually empty
    if remove_empty:
        keep = (new_bbox[:, spatial_dims] >= new_bbox[:, 0]+1-TO_REMOVE) & (new_bbox[:, 1+spatial_dims] >= new_bbox[:, 1]+1-TO_REMOVE)
        if spatial_dims == 3:
            keep = keep & (new_bbox[:, 2+spatial_dims] >= new_bbox[:, 2]+1-TO_REMOVE)
        new_bbox = new_bbox[keep]

    return new_bbox, keep

def box_clip_to_image(
    bbox: torch.Tensor,
    image_size: Union[Sequence[int], torch.Tensor, np.ndarray],
    remove_empty: bool = True,
) -> dict:
    """
    This function makes sure the bounding boxes are within the image.
    Args:
        remove_empty: whether to remove the boxes that are actually empty
    Returns:
        updated box
    """
    spatial_dims = get_dimension(bbox=bbox, image_size=image_size)
    image_box = [0]*spatial_dims+convert_to_list(image_size)
    return box_clip_to_patch(bbox, image_box, remove_empty)

def box_area(bbox: torch.Tensor) -> torch.tensor:
    """
    This function computes the area of each box
    Returns:
        area: 1-D tensor
    """

    spatial_dims = get_dimension(bbox=bbox)

    area = bbox[:, spatial_dims] - bbox[:, 0] + TO_REMOVE
    for axis in range(1, spatial_dims):
        area = area * (bbox[:, axis + spatial_dims] - bbox[:, axis] + TO_REMOVE)

    if torch.isnan(area).any() or torch.isinf(area).any():
        if area.dtype is torch.float16:
            raise ValueError(
                "Box area is NaN or Inf. torch.float16 is used. Please change to torch.float32 and test it again."
            )
        else:
            raise ValueError("Box area is NaN or Inf.")
    return area



def box_iou(bbox1: torch.Tensor, bbox2: torch.Tensor):
    """
    Compute the intersection over union of two set of boxes. This function is not differentialable.

    IMPORTANT: Please run box_clip_to_image(bbox, image_size, mode, remove_empty=True) before computing IoU

    Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    with slight modifications.

    Arguments:
      bbox1: Nx4 or Nx6, make sure they are non-empty
      bbox2: Mx4 or Mx6, make sure they are non-empty

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """

    if bbox1.shape[0] == 0 or bbox2.shape[0] == 0:
        raise ValueError(f"Input of box_iou cannot be empty. Got bbox1.shape={bbox1.shape}, bbox2.shape={bbox12.shape}.")

    spatial_dims = get_dimension(bbox=bbox1)

    # we do computation with compute_dtype to avoid overflow
    box_dtype = bbox1.dtype
    compute_dtype = torch.float32

    # compute area with float32
    area1 = box_area(bbox=bbox1.to(dtype=compute_dtype))  # Nx1
    area2 = box_area(bbox=bbox2.to(dtype=compute_dtype))  # Mx1

    # get the left top and right bottom points for the NxM combinations
    lt = torch.max(bbox1[:, None, :spatial_dims], bbox2[:, :spatial_dims]).to(dtype=compute_dtype)  # [N,M,spatial_dims] left top
    rb = torch.min(bbox1[:, None, spatial_dims:], bbox2[:, spatial_dims:]).to(dtype=compute_dtype)  # [N,M,spatial_dims] right bottom
    # compute size for the intersection region for the NxM combinations
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,spatial_dims]
    inter = wh[:, :, 0]  # [N,M]
    for axis in range(1, spatial_dims):
        inter = inter * wh[:, :, axis]


    # compute IoU and convert back to original box_dtype
    iou = inter / (area1[:, None] + area2 - inter + torch.finfo(compute_dtype).eps)  # [N,M,spatial_dims]
    iou = iou.to(dtype=box_dtype)

    if torch.isnan(iou).any() or torch.isinf(iou).any():
        raise ValueError("Box IoU is NaN or Inf.")

    return iou

def box_giou(bbox1: torch.Tensor, bbox2: torch.Tensor):
    """
    Compute the generalized intersection over union of two set of boxes. This function is not differentialable.

    IMPORTANT: Please run box_clip_to_image(bbox, image_size, mode, remove_empty=True) before computing IoU

    Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    with slight modifications.

    Arguments:
      bbox1: Nx4 or Nx6, make sure they are non-empty
      bbox2: Mx4 or Mx6, make sure they are non-empty

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if bbox1.shape[0] == 0 or bbox2.shape[0] == 0:
        raise ValueError(f"Input of box_giou cannot be empty. Got bbox1.shape={bbox1.shape}, bbox2.shape={bbox12.shape}.")

    spatial_dims = get_dimension(bbox=bbox1)

    # we do computation with compute_dtype to avoid overflow
    box_dtype = bbox1.dtype
    compute_dtype = torch.float32

    # compute area with float32
    area1 = box_area(bbox=bbox1.to(dtype=compute_dtype))  # Nx1
    area2 = box_area(bbox=bbox2.to(dtype=compute_dtype))  # Mx1


    # get the left top and right bottom points for the NxM combinations
    lt = torch.max(bbox1[:, None, :spatial_dims], bbox2[:, :spatial_dims]).to(dtype=compute_dtype)  # [N,M,spatial_dims] left top
    rb = torch.min(bbox1[:, None, spatial_dims:], bbox2[:, spatial_dims:]).to(dtype=compute_dtype)  # [N,M,spatial_dims] right bottom
    # compute size for the intersection region for the NxM combinations
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,spatial_dims]
    inter = wh[:, :, 0]  # [N,M]
    for axis in range(1, spatial_dims):
        inter = inter * wh[:, :, axis]


    # compute IoU and convert back to original box_dtype
    union = area1[:, None] + area2 - inter
    iou = inter / (union+ torch.finfo(compute_dtype).eps)  # [N,M,spatial_dims]

    # enclosure
    lt = torch.min(bbox1[:, None, :spatial_dims], bbox2[:, :spatial_dims]).to(dtype=compute_dtype)  # [N,M,spatial_dims] left top
    rb = torch.max(bbox1[:, None, spatial_dims:], bbox2[:, spatial_dims:]).to(dtype=compute_dtype)  # [N,M,spatial_dims] right bottom
    # compute size for the intersection region for the NxM combinations
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,spatial_dims]
    enclosure = wh[:, :, 0]  # [N,M]
    for axis in range(1, spatial_dims):
        enclosure = enclosure * wh[:, :, axis]

    giou = iou - (enclosure-union)/(enclosure+torch.finfo(compute_dtype).eps)
    giou = giou.to(dtype=box_dtype)
    if torch.isnan(giou).any() or torch.isinf(giou).any():
        raise ValueError("Box GIoU is NaN or Inf.")

    return giou

def box_pair_giou(bbox1: torch.Tensor, bbox2: torch.Tensor):
    """
    Compute the generalized intersection over union of two set of boxes. This function is not differentialable.

    IMPORTANT: Please run box_clip_to_image(bbox, image_size, mode, remove_empty=True) before computing IoU

    Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    with slight modifications.

    Arguments:
      bbox1: Nx4 or Nx6, make sure they are non-empty
      bbox2: Nx4 or Nx6, make sure they are non-empty

    Returns:
      (tensor) iou, sized [N].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """

    if bbox1.shape[0]!=bbox2.shape[0]:
        raise ValueError("bbox1 and bbox2 should be paired.")

    spatial_dims = get_dimension(bbox=bbox1)

    # we do computation with compute_dtype to avoid overflow
    box_dtype = bbox1.dtype
    compute_dtype = torch.float32

    # compute area
    area1 = box_area(bbox=bbox1.to(dtype=compute_dtype))  # Nx1
    area2 = box_area(bbox=bbox2.to(dtype=compute_dtype))  # Nx1

    if cpubool:
        # we do computation on cpu to save gpu memory
        area1 = area1.cpu()
        area2 = area2.cpu()

    # get the left top and right bottom points for the NxM combinations
    lt = torch.max(bbox1[:, :spatial_dims], bbox2[:, :spatial_dims]).to(dtype=compute_dtype)  # [N,spatial_dims] left top
    rb = torch.min(bbox1[:, spatial_dims:], bbox2[:, spatial_dims:]).to(dtype=compute_dtype)  # [N,spatial_dims] right bottom
    if cpubool:
        lt = lt.cpu()
        rb = rb.cpu()
    # compute size for the intersection region for the NxM combinations
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,spatial_dims]
    inter = wh[:, 0]  # [N,M]
    for axis in range(1, spatial_dims):
        inter = inter * wh[:, axis]


    # compute IoU and convert back to original box_dtype
    union = area1 + area2 - inter
    iou = inter / (union+ torch.finfo(compute_dtype).eps)  # [N,spatial_dims]

    # enclosure
    lt = torch.min(bbox1[:, :spatial_dims], bbox2[:, :spatial_dims]).to(dtype=compute_dtype)  # [N,spatial_dims] left top
    rb = torch.max(bbox1[:, spatial_dims:], bbox2[:, spatial_dims:]).to(dtype=compute_dtype)  # [N,spatial_dims] right bottom
    # compute size for the intersection region for the NxM combinations
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,spatial_dims]
    enclosure = wh[:, 0]  # [N,M]
    for axis in range(1, spatial_dims):
        enclosure = enclosure * wh[:, axis]

    giou = iou - (enclosure-union)/(enclosure+torch.finfo(compute_dtype).eps)
    giou = giou.to(dtype=box_dtype) # [N,spatial_dims]
    if torch.isnan(giou).any() or torch.isinf(giou).any():
        raise ValueError("Box GIoU is NaN or Inf.")

    return giou

def non_max_suppression(bbox: torch.Tensor, scores: torch.Tensor, nms_thresh: float, max_proposals=-1, box_overlap_metric="iou"):
    # written by Can Zhao, 2019
    # if there are no boxes, return an empty list
    look_up_option(box_overlap_metric, ["iou", "giou"])
    look_up_option(bbox.shape[1], [4, 6]) // 2
    if bbox.shape[0] == 0:
        return []

    if bbox.shape[0] != scores.shape[0]:
        raise ValueError(f"bbox and scores should have same length, got bbox shape {bbox.shape}, scores shape {scores.shape}")

    scores_sort, indices = torch.sort(deepcopy(scores), descending=True)
    bbox_sort = deepcopy(bbox)[indices, :]

    # initialize the list of picked indexes
    pick = []
    idxs = np.arange(0, bbox_sort.shape[0])
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the first index in the indexes list and add the
        # index value to the list of picked indexes
        i = idxs[0]
        pick.append(i)
        if len(pick) >= max_proposals >= 1:
            break

        # compute the IoU
        if box_overlap_metric=="giou":
            iou = box_giou(bbox_sort[idxs[1:], :], bbox_sort[i:i+1, :])
        else:
            iou = box_iou(bbox_sort[idxs[1:], :], bbox_sort[i:i+1, :])

        # delete all indexes from the index list that have
        idxs = np.delete( idxs, np.concatenate(([0], 1+np.where(iou.cpu().numpy() > nms_thresh)[0]) ) )

    # return only the bounding boxes that were picked using the
    # integer data type
    return indices[pick]


def box_center(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute center point of boxes
    Args:
        boxes: bounding boxes (x1, y1, z1, x2, y2, z2) [N, dims * 2]
    Returns:
        Tensor: center points [N, dims]
    """
    spatial_dims = boxes.shape[1]//2
    centers = [(boxes[:, axis+spatial_dims] + boxes[:, axis]) / 2. for axis in range(spatial_dims)]
    return torch.stack(centers, dim=1)

def box_center_dist(boxes1: torch.Tensor, boxes2: torch.Tensor, euclidean: bool = True) -> \
        Sequence[torch.Tensor]:
    """
    Distance of center points between two sets of boxes
    Arguments:
        boxes1: boxes; (x1, y1, z1, x2, y2, z2)[N, dim * 2]
        boxes2: boxes; (x1, y1, z1, x2, y2, z2)[M, dim * 2]
        euclidean: computed the euclidean distance otherwise it uses the l1
            distance
    Returns:
        Tensor: the NxM matrix containing the pairwise
            distances for every element in boxes1 and boxes2; [N, M]
        Tensor: center points of boxes1
        Tensor: center points of boxes2
    """
    center1 = box_center(boxes1)  # [N, dims]
    center2 = box_center(boxes2)  # [M, dims]

    if euclidean:
        dists = (center1[:, None] - center2[None]).pow(2).sum(-1).sqrt()
    else:
        # before sum: [N, M, dims]
        dists = (center1[:, None] - center2[None]).sum(-1)
    return dists, center1, center2


def center_in_boxes(center: torch.Tensor, boxes: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """
    Checks which center points are within boxes
    Args:
        center: center points [N, dims]
        boxes: boxes [N, dims * 2]
        eps: minimum distance to boarder of boxes
    Returns:
        Tensor: boolean array indicating which center points are within
            the boxes [N]
    """
    spatial_dims = boxes.shape[1]//2
    axes = [center[:, axis] - boxes[:, axis] for axis in range(spatial_dims)] + [boxes[:, axis+spatial_dims] - center[:, axis] for axis in range(spatial_dims)]
    return torch.stack(axes, dim=1).min(dim=1)[0] > eps

def resize_boxes(bbox: torch.Tensor, original_size: List[int], new_size: List[int]) -> torch.Tensor:
    # modified from torchvision
    if len(original_size) != len(new_size):
        raise ValueError("The dimension of original image size should equal to the new image size")
    spatial_dims = get_dimension(bbox, original_size)

    ratios = [
        torch.tensor(s, dtype=bbox.dtype, device=bbox.device)
        / torch.tensor(s_orig, dtype=bbox.dtype, device=bbox.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    box_xxyy = split_into_corners(deepcopy(bbox))

    for axis in range(spatial_dims):
        box_xxyy[2 * axis] = box_xxyy[2 * axis] * ratios[axis]
        box_xxyy[2 * axis + 1] = box_xxyy[2 * axis + 1] * ratios[axis]

    box_xxyy = torch.stack(box_xxyy, dim=1)

    return box_convert_standard_mode(box_xxyy, mode=XXYYZZ_MODE[spatial_dims-2])
