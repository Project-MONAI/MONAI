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

from monai.config.type_definitions import NdarrayOrTensor

# from monai.utils.misc import ensure_tuple_rep
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_to_numpy, convert_to_tensor

CORNER_CORNER_MODE = ["xyxy", "xyzxyz"]  # [2d_mode, 3d_mode]
XXYYZZ_MODE = ["xxyy", "xxyyzz"]  # [2d_mode, 3d_mode]
CORNER_SIZE_MODE = ["xywh", "xyzwhd"]  # [2d_mode, 3d_mode]
CENTER_SIZE_MODE = ["ccwh", "cccwhd"]  # [2d_mode, 3d_mode]

STANDARD_MODE = CORNER_CORNER_MODE  # [2d_mode, 3d_mode]
SUPPORT_MODE = CORNER_CORNER_MODE + XXYYZZ_MODE + CORNER_SIZE_MODE + CENTER_SIZE_MODE

# TO_REMOVE = 0 if in 'xxyy','xxyyzz' mode, the bottom-right corner is not included in the box,
#      i.e., when x_min=1, x_max=2, we have w = 1
# TO_REMOVE = 1  if in 'xxyy','xxyyzz' mode, the bottom-right corner is included in the box,
#       i.e., when x_min=1, x_max=2, we have w = 2
# Currently only TO_REMOVE = 0 has been tested. Please use TO_REMOVE = 0
TO_REMOVE = 0  # x_max-x_min = w -TO_REMOVE.


def convert_to_list(in_sequence: Union[Sequence, torch.Tensor, np.ndarray]) -> list:
    """
    Convert a torch.Tensor, or np array input to list
    Args:
        in_sequence: Sequence or torch.Tensor or np.ndarray
    Returns:
        in_sequence_list: a list

    """
    in_sequence_list = deepcopy(in_sequence)
    if isinstance(in_sequence, torch.Tensor):
        in_sequence_list = in_sequence_list.detach().cpu().numpy().tolist()
    elif isinstance(in_sequence, np.ndarray):
        in_sequence_list = in_sequence_list.tolist()
    elif not isinstance(in_sequence, list):
        in_sequence_list = list(in_sequence_list)
    return in_sequence_list


def get_dimension(
    bbox: Union[torch.Tensor, np.ndarray, None] = None,
    image_size: Union[Sequence[int], torch.Tensor, np.ndarray, None] = None,
    mode: Union[str, None] = None,
) -> int:
    """
    Get spatial dimension for the giving setting.
    Missing input is allowed. But at least one of the input value should be given.
    Args:
        bbox: bounding box, Nx4 or Nx6 torch tensor or ndarray
        image_size: Length of 2 or 3. Data format is list, or np.ndarray, or tensor of int
        mode: box mode, choose from SUPPORT_MODE. If mode is not given, this func will assume mode is STANDARD_MODE
    Returns:
        spatial_dimension: 2 or 3

    Example:
        bbox = torch.zeros(10,6)
        get_dimension(bbox, mode="xyzxyz") will return 3
        get_dimension(bbox, mode="xyzxyz", image_size=[100,200,200]) will return 3
        get_dimension(mode="xyzxyz") will return 3
    """
    spatial_dims_set = set()
    if image_size is not None:
        spatial_dims_set.add(len(image_size))
    if mode is not None:
        spatial_dims_set.add(int(len(mode) / 2))
    if bbox is not None:
        spatial_dims_set.add(int(bbox.shape[1] / 2))
    spatial_dims_list = list(spatial_dims_set)
    if len(spatial_dims_list) == 0:
        raise ValueError("At least one of bbox, image_size, and mode needs to be non-empty.")
    elif len(spatial_dims_list) == 1:
        spatial_dims = int(spatial_dims_list[0])
        spatial_dims = look_up_option(spatial_dims, supported=[2, 3])
        return int(spatial_dims)
    else:
        raise ValueError("The dimension of bbox, image_size, mode should match with each other.")


def get_standard_mode(spatial_dims: int) -> str:
    """
    Get the mode name for the given spatial dimension
    Args:
        spatial_dims: 2 or 3
    Returns:
        mode name, choose from STANDARD_MODE

    Example:
        get_standard_mode(spatial_dims = 2)

    """
    if spatial_dims == 2:
        return STANDARD_MODE[0]
    elif spatial_dims == 3:
        return STANDARD_MODE[1]
    else:
        raise ValueError(f"Images should have 2 or 3 dimensions, got {spatial_dims}")


def split_into_corners(bbox: NdarrayOrTensor, mode: Union[str, None] = None):
    """
    This internal function outputs the corner coordinates of the bbox
    Args:
        bbox: bounding box, Nx4 or Nx6 torch tensor or ndarray
        mode: box mode, choose from SUPPORT_MODE. If mode is not given, this func will assume mode is STANDARD_MODE
    Returns:
        if 2D image, outputs (xmin, xmax, ymin, ymax)
        if 3D images, outputs (xmin, xmax, ymin, ymax, zmin, zmax)
        xmin for example, is a Nx1 tensor
    
    Example:
        bbox = torch.zeros(10,6)
        split_into_corners(bbox, mode="cccwhd")
    """
    # convert numpy to tensor if needed
    if isinstance(bbox, np.ndarray):
        bbox = convert_to_tensor(bbox)
        numpy_bool = True
    else:
        numpy_bool = False

    # convert to float32 when computing torch.clamp, which does not support float16
    box_dtype = bbox.dtype
    compute_dtype = torch.float32

    if mode is None:
        mode = get_standard_mode(int(bbox.shape[1] / 2))
    mode = look_up_option(mode, supported=SUPPORT_MODE)

    # split tensor into corners
    if mode in ["xxyy", "xxyyzz"]:
        split_result = bbox.split(1, dim=-1)
    elif mode == "xyzxyz":
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.split(1, dim=-1)
        split_result = (xmin, xmax, ymin, ymax, zmin, zmax)
    elif mode == "xyxy":
        xmin, ymin, xmax, ymax = bbox.split(1, dim=-1)
        split_result = (xmin, xmax, ymin, ymax)
    elif mode == "xyzwhd":
        xmin, ymin, zmin, w, h, d = bbox.split(1, dim=-1)
        split_result = (
            xmin,
            xmin + (w - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
            ymin,
            ymin + (h - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
            zmin,
            zmin + (d - TO_REMOVE).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
        )
    elif mode == "xywh":
        xmin, ymin, w, h = bbox.split(1, dim=-1)
        split_result = (xmin, xmin + (w - TO_REMOVE).clamp(min=0), ymin, ymin + (h - TO_REMOVE).clamp(min=0))
    elif mode == "cccwhd":
        xc, yc, zc, w, h, d = bbox.split(1, dim=-1)
        split_result = (
            xc - ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
            xc + ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
            yc - ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
            yc + ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
            zc - ((d - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
            zc + ((d - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
        )
    elif mode == "ccwh":
        xc, yc, w, h = bbox.split(1, dim=-1)
        split_result = (
            xc - ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
            xc + ((w - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
            yc - ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
            yc + ((h - TO_REMOVE) / 2.0).to(dtype=compute_dtype).clamp(min=0).to(dtype=box_dtype),
        )
    else:
        raise RuntimeError("Should not be here")

    # convert tensor back to numpy if needed
    if numpy_bool:
        split_result = convert_to_numpy(split_result)
    return split_result


def box_convert_mode(
    bbox1: NdarrayOrTensor, mode1: Union[str, None] = None, mode2: Union[str, None] = None
) -> NdarrayOrTensor:
    """
    This function converts the bbox1 in mode 1 to the mode2
    Args:
        bbox1: source bounding box, Nx4 or Nx6 torch tensor or ndarray
        mode1: source box mode, choose from SUPPORT_MODE. If mode is not given, this func will assume mode is STANDARD_MODE
        mode2: target box mode, choose from SUPPORT_MODE. If mode is not given, this func will assume mode is STANDARD_MODE
    Returns:
        bbox2: bounding box with target mode, does not share memory with original bbox1

    Example:
        bbox = torch.zeros(10,6)
        box_convert_mode(bbox1=bbox, mode1="xyzxyz", mode2="cccwhd")
    """

    # convert numpy to tensor if needed
    if isinstance(bbox1, np.ndarray):
        bbox1 = convert_to_tensor(bbox1)
        numpy_bool = True
    else:
        numpy_bool = False

    # check whether the bbox and the new mode is valid
    if mode1 is None:
        mode1 = get_standard_mode(int(bbox1.shape[1] / 2))
    if mode2 is None:
        mode2 = get_standard_mode(int(bbox1.shape[1] / 2))
    mode1 = look_up_option(mode1, supported=SUPPORT_MODE)
    mode2 = look_up_option(mode2, supported=SUPPORT_MODE)

    spatial_dims = get_dimension(bbox=bbox1, mode=mode1)
    if len(mode1) != len(mode2):
        raise ValueError("The dimension of the new mode should have the same spatial dimension as the old mode.")

    # if mode not changed, return original box
    if mode1 == mode2:
        bbox2 = deepcopy(bbox1)
    # convert mode for bbox
    elif mode2 in ["xxyy", "xxyyzz"]:
        corners = split_into_corners(bbox1, mode1)
        bbox2 = torch.cat(corners, dim=-1)
    else:
        if spatial_dims == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = split_into_corners(bbox1, mode1)
            if mode2 == "xyzxyz":
                bbox2 = torch.cat((xmin, ymin, zmin, xmax, ymax, zmax), dim=-1)
            elif mode2 == "xyzwhd":
                bbox2 = torch.cat(
                    (xmin, ymin, zmin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE, zmax - zmin + TO_REMOVE),
                    dim=-1,
                )
            elif mode2 == "cccwhd":
                bbox2 = torch.cat(
                    (
                        (xmin + xmax + TO_REMOVE) / 2,
                        (ymin + ymax + TO_REMOVE) / 2,
                        (zmin + zmax + TO_REMOVE) / 2,
                        xmax - xmin + TO_REMOVE,
                        ymax - ymin + TO_REMOVE,
                        zmax - zmin + TO_REMOVE,
                    ),
                    dim=-1,
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
                        (xmin + xmax + TO_REMOVE) / 2,
                        (ymin + ymax + TO_REMOVE) / 2,
                        xmax - xmin + TO_REMOVE,
                        ymax - ymin + TO_REMOVE,
                    ),
                    dim=-1,
                )
            else:
                raise ValueError("We support only bbox mode in " + str(SUPPORT_MODE) + f", got {mode2}")
        else:
            raise ValueError(f"Images should have 2 or 3 dimensions, got {spatial_dims}")

    # convert tensor back to numpy if needed
    if numpy_bool:
        bbox2 = convert_to_numpy(bbox2)

    return bbox2


def box_convert_standard_mode(bbox: NdarrayOrTensor, mode: Union[str, None] = None) -> NdarrayOrTensor:
    """
    Convert given bbox to standard mode
    Args:
        bbox: source bounding box, Nx4 or Nx6 torch tensor or ndarray
        mode: source box mode, choose from SUPPORT_MODE. If mode is not given, this func will assume mode is STANDARD_MODE
    Returns:
        bbox2: bounding box with standard mode, does not share memory with original bbox1

    Example:
        bbox = torch.zeros(10,6)
        box_convert_mode(bbox=bbox, mode="xxyyzz")
    """
    if mode is None:
        mode = get_standard_mode(int(bbox.shape[1] / 2))
    mode = look_up_option(mode, supported=SUPPORT_MODE)
    spatial_dims = get_dimension(bbox=bbox, mode=mode)
    mode_standard = get_standard_mode(spatial_dims)
    return box_convert_mode(bbox1=bbox, mode1=mode, mode2=mode_standard)


# def point_interp(
#     point: NdarrayOrTensor, zoom: Union[Sequence[float], float]
# ) -> Union[Sequence, torch.Tensor, np.ndarray]:
#     """
#     Convert point position from one pixel/voxel size to another pixel/voxel size
#     Args:
#         point: point coordinate, Nx2 or Nx3, [x, y] or [x, y, z]
#         zoom: The zoom factor along the spatial axes.
#             If a float, zoom is the same for each spatial axis.
#             If a sequence, zoom should contain one value for each spatial axis.
#     Returns:
#         point2: zoomed point coordinate, does not share memory with original point
#     """
#     # make sure the spatial dimensions of the inputs match with each other
#     spatial_dims = point.shape[1]
#     spatial_dims = look_up_option(spatial_dims, supported=[2, 3])

#     # compute new point
#     point2 = deepcopy(point)
#     _zoom = ensure_tuple_rep(zoom, spatial_dims)
#     for axis in range(0, spatial_dims):
#         point2[:, axis] = point[:, axis] * _zoom[axis]
#     return point2


# def box_interp(
#     bbox: NdarrayOrTensor, zoom: Union[Sequence[float], float], mode: Union[str, None] = None
# ) -> torch.Tensor:
#     """
#     Interpolate bbox
#     Args:
#         bbox: bounding box, Nx4 or Nx6 torch tensor or ndarray
#         mode: box mode, choose from SUPPORT_MODE. If mode is not given, this func will assume mode is STANDARD_MODE
#         zoom: The zoom factor along the spatial axes.
#             If a float, zoom is the same for each spatial axis.
#             If a sequence, zoom should contain one value for each spatial axis.

#     Returns:
#         returned interpolated bbox has the same mode as bbox, does not share memory with original bbox
#     """
#     if mode is None:
#         mode = get_standard_mode(int(bbox.shape[1] / 2))
#     mode = look_up_option(mode, supported=SUPPORT_MODE)
#     spatial_dims = get_dimension(bbox=bbox, mode=mode)

#     # convert to standard mode
#     mode_standard = get_standard_mode(spatial_dims)
#     bbox_standard = box_convert_mode(bbox1=bbox, mode1=mode, mode2=mode_standard)

#     # interp
#     corner_lt = point_interp(bbox_standard[:, :spatial_dims], zoom)
#     corner_rb = point_interp(bbox_standard[:, spatial_dims:], zoom)

#     bbox_standard_interp = deepcopy(bbox_standard)
#     bbox_standard_interp[:, :spatial_dims] = corner_lt
#     bbox_standard_interp[:, spatial_dims:] = corner_rb

#     # convert back
#     bbox2 = box_convert_mode(bbox1=bbox_standard_interp, mode1=mode_standard, mode2=mode)
#     return bbox2

# def point_affine(
#     point: NdarrayOrTensor, affine: NdarrayOrTensor, include_shift: bool = True
# ) -> Union[Sequence, torch.Tensor, np.ndarray]:
#     """
#     Convert point position from one pixel/voxel size to another pixel/voxel size
#     Args:
#         point: point coordinate, Nx2 or Nx3, [x, y] or [x, y, z]
#         affine: affine transform
#         include_shift: does the func apply translation (shift) in the affine transform
#     Returns:
#         point2: transformed point coordinate, does not share memory with original point
#     """
#     # make sure the spatial dimensions of the inputs match with each other
#     spatial_dims = point.shape[1]
#     spatial_dims = look_up_option(spatial_dims, supported=[2, 3])

#     # convert numpy to tensor if needed
#     if isinstance(point, np.ndarray):
#         point = convert_to_tensor(point)
#         numpy_bool = True
#     else:
#         numpy_bool = False
#     affine = convert_to_tensor(affine, device=point.device, dtype=point.dtype)

#     # compute new point
#     if include_shift:
#         # append 1 to form Nx(spatial_dims+1) vector, then transpose
#         point2 = torch.cat(
#             [point, torch.ones(point.shape[0], 1, device=point.device, dtype=point.dtype)], dim=1
#         ).transpose(0, 1)
#         # apply affine
#         point2 = torch.matmul(affine, point2)
#         # remove appended 1 and transpose back
#         point2 = point2[:spatial_dims, :].transpose(0, 1)
#     else:
#         point2 = point.transpose(0, 1)
#         point2 = torch.matmul(affine[:spatial_dims, :spatial_dims], point2)
#         point2 = point2.transpose(0, 1)

#     # convert tensor back to numpy if needed
#     if numpy_bool:
#         point2 = convert_to_numpy(point2)
#     return point2


# def box_affine(bbox: NdarrayOrTensor, affine: NdarrayOrTensor, mode: Union[str, None] = None) -> torch.Tensor:
#     """
#     This function applys affine matrixs to the bbox
#     Args:
#         bbox: bounding box, Nx4 or Nx6 torch tensor or ndarray
#         affine: affine matric to be applied to the box coordinate, (spatial_dims+1)x(spatial_dims+1)
#         mode: box mode, choose from SUPPORT_MODE. If mode is not given, this func will assume mode is STANDARD_MODE
#     Returns:
#         returned affine transformed bbox has the same mode as bbox, does not share memory with original bbox
#     """
#     # convert numpy to tensor if needed
#     if isinstance(bbox, np.ndarray):
#         bbox = convert_to_tensor(bbox)
#         numpy_bool = True
#     else:
#         numpy_bool = False

#     box_dtype = bbox.dtype
#     compute_dtype = torch.float32
#     if box_dtype is torch.float16:
#         bbox = bbox.to(dtype=compute_dtype)
#     affine = convert_to_tensor(affine, device=bbox.device, dtype=bbox.dtype)

#     if mode is None:
#         mode = get_standard_mode(int(bbox.shape[1] / 2))
#     mode = look_up_option(mode, supported=SUPPORT_MODE)
#     spatial_dims = get_dimension(bbox=bbox, mode=mode)

#     if mode in ["xxyy", "xxyyzz", "xyxy", "xyzxyz"]:
#         # extract left top and right bottom, and apply affine
#         if mode in ["xxyy", "xxyyzz"]:
#             lt = point_affine(bbox[:, ::2], affine, include_shift=True)
#             rb = point_affine(bbox[:, 1::2], affine, include_shift=True)
#         if mode in ["xyxy", "xyzxyz"]:
#             lt = point_affine(bbox[:, :spatial_dims], affine, include_shift=True)
#             rb = point_affine(bbox[:, spatial_dims:], affine, include_shift=True)

#         lt_new, _ = torch.min(torch.stack([lt, rb], dim=2), dim=2)
#         rb_new, _ = torch.max(torch.stack([lt, rb], dim=2), dim=2)

#         bbox2 = box_convert_mode(torch.cat([lt_new, rb_new], dim=1), mode1=STANDARD_MODE[spatial_dims - 2], mode2=mode)

#     elif mode in ["ccwh", "cccwhd", "xywh", "xyzwhd"]:
#         cc = point_affine(bbox[:, :spatial_dims], affine, include_shift=True)
#         wh = point_affine(bbox[:, spatial_dims:], affine, include_shift=False).absolute()
#         bbox2 = torch.cat([cc, wh], dim=1)

#     else:
#         raise RuntimeError("Should not be here")

#     # convert tensor back to numpy if needed
#     if numpy_bool:
#         bbox2 = convert_to_numpy(bbox2.to(dtype=box_dtype))
#     return bbox2


# def box_clip_to_patch(
#     bbox: NdarrayOrTensor, patch_box: Union[Sequence[int], torch.Tensor, np.ndarray], remove_empty: bool = True
# ):
#     """
#     This function makes sure the bounding boxes are within the patch.
#     Args:
#         bbox: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#         patch_box: The coordinate of the target patch to clip, it follows standard mode
#         remove_empty: whether to remove the boxes that are actually empty
#     Returns:
#         new_bbox: updated box, does not share memory with original bbox
#         keep: the indice of the new_bbox regarding to input bbox. When remove_empty=True, only some of the boxes are kept
#     """
#     if bbox.shape[0] == 0:
#         return deepcopy(bbox), []

#     spatial_dims = get_dimension(bbox=bbox)
#     new_bbox = deepcopy(bbox)

#     # convert numpy to tensor if needed
#     if isinstance(new_bbox, np.ndarray):
#         new_bbox = convert_to_tensor(new_bbox)
#         numpy_bool = True
#     else:
#         numpy_bool = False

#     # convert to float32 since torch.clamp_ does not support float16
#     box_dtype = new_bbox.dtype
#     compute_dtype = torch.float32
#     if box_dtype is torch.float16:
#         new_bbox = new_bbox.to(dtype=compute_dtype)

#     # makes sure the bounding boxes are within the image
#     for axis in range(0, spatial_dims):
#         new_bbox[:, axis].clamp_(min=patch_box[axis], max=patch_box[axis + spatial_dims] - TO_REMOVE)
#         new_bbox[:, axis + spatial_dims].clamp_(min=patch_box[axis], max=patch_box[axis + spatial_dims] - TO_REMOVE)
#         new_bbox[:, axis] -= patch_box[axis]
#         new_bbox[:, axis + spatial_dims] -= patch_box[axis]

#     # remove the boxes that are actually empty
#     if remove_empty:
#         keep = (new_bbox[:, spatial_dims] >= new_bbox[:, 0] + 1 - TO_REMOVE) & (
#             new_bbox[:, 1 + spatial_dims] >= new_bbox[:, 1] + 1 - TO_REMOVE
#         )
#         if spatial_dims == 3:
#             keep = keep & (new_bbox[:, 2 + spatial_dims] >= new_bbox[:, 2] + 1 - TO_REMOVE)
#         new_bbox = new_bbox[keep]

#     # convert tensor back to numpy if needed
#     new_bbox = new_bbox.to(dtype=box_dtype)
#     if numpy_bool:
#         new_bbox = convert_to_numpy(new_bbox)

#     return new_bbox, keep


# def box_clip_to_image(
#     bbox: NdarrayOrTensor, image_size: Union[Sequence[int], torch.Tensor, np.ndarray], remove_empty: bool = True
# ):
#     """
#     This function makes sure the bounding boxes are within the image.
#     Args:
#         bbox: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#         remove_empty: whether to remove the boxes that are actually empty
#     Returns:
#         updated box
#     """
#     spatial_dims = get_dimension(bbox=bbox, image_size=image_size)
#     image_box = [0] * spatial_dims + convert_to_list(image_size)
#     return box_clip_to_patch(bbox, image_box, remove_empty)


# def box_area(bbox: NdarrayOrTensor) -> NdarrayOrTensor:
#     """
#     This function computes the area of each box
#     Args:
#         bbox: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#     Returns:
#         area: 1-D tensor
#     """

#     spatial_dims = get_dimension(bbox=bbox)

#     area = bbox[:, spatial_dims] - bbox[:, 0] + TO_REMOVE
#     for axis in range(1, spatial_dims):
#         area = area * (bbox[:, axis + spatial_dims] - bbox[:, axis] + TO_REMOVE)

#     if isinstance(area, np.ndarray):
#         area = convert_to_tensor(area)
#         numpy_bool = True
#     else:
#         numpy_bool = False

#     if area.isnan().any() or area.isinf().any():
#         if area.dtype is torch.float16:
#             raise ValueError("Box area is NaN or Inf. bbox is float16. Please change to float32 and test it again.")
#         else:
#             raise ValueError("Box area is NaN or Inf.")

#     if numpy_bool:
#         area = convert_to_numpy(area)
#     return area


# def box_iou(bbox1: NdarrayOrTensor, bbox2: NdarrayOrTensor) -> NdarrayOrTensor:
#     """
#     Compute the intersection over union of two set of boxes. This function is not differentialable.

#     IMPORTANT: Please run box_clip_to_image(bbox, image_size, mode, remove_empty=True) before computing IoU

#     Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
#     with slight modifications.

#     Args:
#         bbox1: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#         bbox2: bounding box, Mx4 or Mx6 torch tensor. The box mode is assumed to be STANDARD_MODE

#     Returns:
#       (tensor) iou, sized [N,M].

#     Reference:
#       https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
#     """

#     # convert numpy to tensor if needed
#     if isinstance(bbox1, np.ndarray):
#         bbox1 = convert_to_tensor(bbox1)
#         bbox2 = convert_to_tensor(bbox2)
#         numpy_bool = True
#     else:
#         numpy_bool = False

#     spatial_dims = get_dimension(bbox=bbox1)

#     # we do computation with compute_dtype to avoid overflow
#     box_dtype = bbox1.dtype
#     compute_dtype = torch.float32

#     # compute area with float32
#     area1 = box_area(bbox=bbox1.to(dtype=compute_dtype))  # Nx1
#     area2 = box_area(bbox=bbox2.to(dtype=compute_dtype))  # Mx1

#     # get the left top and right bottom points for the NxM combinations
#     lt = torch.max(bbox1[:, None, :spatial_dims], bbox2[:, :spatial_dims]).to(
#         dtype=compute_dtype
#     )  # [N,M,spatial_dims] left top
#     rb = torch.min(bbox1[:, None, spatial_dims:], bbox2[:, spatial_dims:]).to(
#         dtype=compute_dtype
#     )  # [N,M,spatial_dims] right bottom
#     # compute size for the intersection region for the NxM combinations
#     wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,spatial_dims]
#     inter = wh[:, :, 0]  # [N,M]
#     for axis in range(1, spatial_dims):
#         inter = inter * wh[:, :, axis]

#     # compute IoU and convert back to original box_dtype
#     iou = inter / (area1[:, None] + area2 - inter + torch.finfo(compute_dtype).eps)  # [N,M,spatial_dims]
#     iou = iou.to(dtype=box_dtype)

#     if torch.isnan(iou).any() or torch.isinf(iou).any():
#         raise ValueError("Box IoU is NaN or Inf.")

#     # convert tensor back to numpy if needed
#     if numpy_bool:
#         iou = convert_to_numpy(iou)
#     return iou


# def box_giou(bbox1: NdarrayOrTensor, bbox2: NdarrayOrTensor) -> NdarrayOrTensor:
#     """
#     Compute the generalized intersection over union of two set of boxes. This function is not differentialable.

#     IMPORTANT: Please run box_clip_to_image(bbox, image_size, mode, remove_empty=True) before computing IoU

#     Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
#     with slight modifications.

#     Args:
#         bbox1: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#         bbox2: bounding box, Mx4 or Mx6 torch tensor. The box mode is assumed to be STANDARD_MODE

#     Returns:
#       (tensor) iou, sized [N,M].

#     Reference:
#       https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
#     """
#     # convert numpy to tensor if needed
#     if isinstance(bbox1, np.ndarray):
#         bbox1 = convert_to_tensor(bbox1)
#         bbox2 = convert_to_tensor(bbox2)
#         numpy_bool = True
#     else:
#         numpy_bool = False

#     spatial_dims = get_dimension(bbox=bbox1)

#     # we do computation with compute_dtype to avoid overflow
#     box_dtype = bbox1.dtype
#     compute_dtype = torch.float32

#     # compute area with float32
#     area1 = box_area(bbox=bbox1.to(dtype=compute_dtype))  # Nx1
#     area2 = box_area(bbox=bbox2.to(dtype=compute_dtype))  # Mx1

#     # get the left top and right bottom points for the NxM combinations
#     lt = torch.max(bbox1[:, None, :spatial_dims], bbox2[:, :spatial_dims]).to(
#         dtype=compute_dtype
#     )  # [N,M,spatial_dims] left top
#     rb = torch.min(bbox1[:, None, spatial_dims:], bbox2[:, spatial_dims:]).to(
#         dtype=compute_dtype
#     )  # [N,M,spatial_dims] right bottom
#     # compute size for the intersection region for the NxM combinations
#     wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,spatial_dims]
#     inter = wh[:, :, 0]  # [N,M]
#     for axis in range(1, spatial_dims):
#         inter = inter * wh[:, :, axis]

#     # compute IoU and convert back to original box_dtype
#     union = area1[:, None] + area2 - inter
#     iou = inter / (union + torch.finfo(compute_dtype).eps)  # [N,M,spatial_dims]

#     # enclosure
#     lt = torch.min(bbox1[:, None, :spatial_dims], bbox2[:, :spatial_dims]).to(
#         dtype=compute_dtype
#     )  # [N,M,spatial_dims] left top
#     rb = torch.max(bbox1[:, None, spatial_dims:], bbox2[:, spatial_dims:]).to(
#         dtype=compute_dtype
#     )  # [N,M,spatial_dims] right bottom
#     # compute size for the intersection region for the NxM combinations
#     wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,spatial_dims]
#     enclosure = wh[:, :, 0]  # [N,M]
#     for axis in range(1, spatial_dims):
#         enclosure = enclosure * wh[:, :, axis]

#     giou = iou - (enclosure - union) / (enclosure + torch.finfo(compute_dtype).eps)
#     giou = giou.to(dtype=box_dtype)
#     if torch.isnan(giou).any() or torch.isinf(giou).any():
#         raise ValueError("Box GIoU is NaN or Inf.")

#     # convert tensor back to numpy if needed
#     if numpy_bool:
#         giou = convert_to_numpy(giou)
#     return giou


# def box_pair_giou(bbox1: NdarrayOrTensor, bbox2: NdarrayOrTensor) -> NdarrayOrTensor:
#     """
#     Compute the generalized intersection over union of two set of boxes. This function is not differentialable.

#     IMPORTANT: Please run box_clip_to_image(bbox, image_size, mode, remove_empty=True) before computing IoU

#     Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
#     with slight modifications.

#     Args:
#         bbox1: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#         bbox2: bounding box, Mx4 or Mx6 torch tensor. The box mode is assumed to be STANDARD_MODE

#     Returns:
#       (tensor) iou, sized [N].

#     Reference:
#       https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
#     """

#     if bbox1.shape[0] != bbox2.shape[0]:
#         raise ValueError("bbox1 and bbox2 should be paired.")

#     # convert numpy to tensor if needed
#     if isinstance(bbox1, np.ndarray):
#         bbox1 = convert_to_tensor(bbox1)
#         bbox2 = convert_to_tensor(bbox2)
#         numpy_bool = True
#     else:
#         numpy_bool = False

#     spatial_dims = get_dimension(bbox=bbox1)

#     # we do computation with compute_dtype to avoid overflow
#     box_dtype = bbox1.dtype
#     compute_dtype = torch.float32

#     # compute area
#     area1 = box_area(bbox=bbox1.to(dtype=compute_dtype))  # Nx1
#     area2 = box_area(bbox=bbox2.to(dtype=compute_dtype))  # Nx1

#     # get the left top and right bottom points for the NxM combinations
#     lt = torch.max(bbox1[:, :spatial_dims], bbox2[:, :spatial_dims]).to(
#         dtype=compute_dtype
#     )  # [N,spatial_dims] left top
#     rb = torch.min(bbox1[:, spatial_dims:], bbox2[:, spatial_dims:]).to(
#         dtype=compute_dtype
#     )  # [N,spatial_dims] right bottom
#     # compute size for the intersection region for the NxM combinations
#     wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,spatial_dims]
#     inter = wh[:, 0]  # [N,M]
#     for axis in range(1, spatial_dims):
#         inter = inter * wh[:, axis]

#     # compute IoU and convert back to original box_dtype
#     union = area1 + area2 - inter
#     iou = inter / (union + torch.finfo(compute_dtype).eps)  # [N,spatial_dims]

#     # enclosure
#     lt = torch.min(bbox1[:, :spatial_dims], bbox2[:, :spatial_dims]).to(
#         dtype=compute_dtype
#     )  # [N,spatial_dims] left top
#     rb = torch.max(bbox1[:, spatial_dims:], bbox2[:, spatial_dims:]).to(
#         dtype=compute_dtype
#     )  # [N,spatial_dims] right bottom
#     # compute size for the intersection region for the NxM combinations
#     wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,spatial_dims]
#     enclosure = wh[:, 0]  # [N,M]
#     for axis in range(1, spatial_dims):
#         enclosure = enclosure * wh[:, axis]

#     giou = iou - (enclosure - union) / (enclosure + torch.finfo(compute_dtype).eps)
#     giou = giou.to(dtype=box_dtype)  # [N,spatial_dims]
#     if torch.isnan(giou).any() or torch.isinf(giou).any():
#         raise ValueError("Box GIoU is NaN or Inf.")

#     # convert tensor back to numpy if needed
#     if numpy_bool:
#         giou = convert_to_numpy(giou)
#     return giou


# def non_max_suppression(
#     bbox: NdarrayOrTensor, scores: NdarrayOrTensor, nms_thresh: float, max_proposals=-1, box_overlap_metric="iou"
# ):
#     """
#     written by Can Zhao, 2019
#     if there are no boxes, return an empty list
#     Args:
#         bbox: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#     """
#     look_up_option(box_overlap_metric, ["iou", "giou"])
#     look_up_option(bbox.shape[1], [4, 6]) // 2
#     if bbox.shape[0] == 0:
#         return []

#     if bbox.shape[0] != scores.shape[0]:
#         raise ValueError(
#             f"bbox and scores should have same length, got bbox shape {bbox.shape}, scores shape {scores.shape}"
#         )

#     # convert numpy to tensor if needed
#     if isinstance(bbox, np.ndarray):
#         bbox = convert_to_tensor(bbox)
#         scores = convert_to_tensor(scores)
#         numpy_bool = True
#     else:
#         numpy_bool = False

#     scores_sort, indices = torch.sort(scores, descending=True)
#     bbox_sort = deepcopy(bbox)[indices, :]

#     # initialize the list of picked indexes
#     pick = []
#     idxs = np.arange(0, bbox_sort.shape[0])
#     # keep looping while some indexes still remain in the indexes
#     # list
#     while len(idxs) > 0:
#         # grab the first index in the indexes list and add the
#         # index value to the list of picked indexes
#         i = idxs[0]
#         pick.append(i)
#         if len(pick) >= max_proposals >= 1:
#             break

#         # compute the IoU
#         if box_overlap_metric == "giou":
#             iou = box_giou(bbox_sort[idxs[1:], :], bbox_sort[i : i + 1, :])
#         else:
#             iou = box_iou(bbox_sort[idxs[1:], :], bbox_sort[i : i + 1, :])

#         # delete all indexes from the index list that have overlap > nms_thresh
#         idxs = np.delete(idxs, np.concatenate(([0], 1 + np.where(iou.cpu().numpy() > nms_thresh)[0])))

#     # return only the bounding boxes that were picked using the
#     # integer data type
#     pick_idx = indices[pick]
#     # convert tensor back to numpy if needed
#     if numpy_bool:
#         pick_idx = convert_to_numpy(pick_idx)
#     return pick_idx


# def box_center(bbox: NdarrayOrTensor) -> torch.Tensor:
#     """
#     Compute center point of bbox
#     Args:
#         bbox: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#     Returns:
#         Tensor: center points [N, dims]
#     """
#     spatial_dims = bbox.shape[1] // 2
#     centers = [(bbox[:, axis + spatial_dims] + bbox[:, axis]) / 2.0 for axis in range(spatial_dims)]

#     if isinstance(bbox, np.ndarray):
#         return np.stack(centers, axis=1)
#     else:
#         return torch.stack(centers, dim=1)


# def box_center_dist(bbox1: torch.Tensor, bbox2: torch.Tensor, euclidean: bool = True) -> Sequence[torch.Tensor]:
#     """
#     Distance of center points between two sets of bbox
#     Args:
#         bbox1: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#         bbox2: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#         euclidean: computed the euclidean distance otherwise it uses the l1
#             distance
#     Returns:
#         Tensor: the NxM matrix containing the pairwise
#             distances for every element in bbox1 and bbox2; [N, M]
#         Tensor: center points of bbox1
#         Tensor: center points of bbox2
#     """
#     # convert numpy to tensor if needed
#     if isinstance(bbox1, np.ndarray):
#         bbox1 = convert_to_tensor(bbox1)
#         bbox2 = convert_to_tensor(bbox2)
#         numpy_bool = True
#     else:
#         numpy_bool = False

#     box_dtype = bbox1.dtype
#     compute_dtype = torch.float32

#     center1 = box_center(bbox1.to(compute_dtype))  # [N, dims]
#     center2 = box_center(bbox2.to(compute_dtype))  # [M, dims]

#     if euclidean:
#         dists = (center1[:, None] - center2[None]).pow(2).sum(-1).sqrt()
#     else:
#         # before sum: [N, M, dims]
#         dists = (center1[:, None] - center2[None]).sum(-1)

#     # convert tensor back to numpy if needed
#     dists, center1, center2 = dists.to(box_dtype), center1.to(box_dtype), center2.to(box_dtype)
#     if numpy_bool:
#         dists, center1, center2 = convert_to_numpy(dists), convert_to_numpy(center1), convert_to_numpy(center2)
#     return dists, center1, center2


# def center_in_boxes(center: NdarrayOrTensor, bbox: NdarrayOrTensor, eps: float = 0.01) -> NdarrayOrTensor:
#     """
#     Checks which center points are within bbox
#     Args:
#         bbox: bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#         center: center points [N, dims]
#         eps: minimum distance to boarder of bbox
#     Returns:
#         Tensor: boolean array indicating which center points are within
#             the bbox [N]
#     """
#     spatial_dims = bbox.shape[1] // 2
#     axes = [center[:, axis] - bbox[:, axis] for axis in range(spatial_dims)] + [
#         bbox[:, axis + spatial_dims] - center[:, axis] for axis in range(spatial_dims)
#     ]
#     if isinstance(bbox, np.ndarray):
#         return np.stack(axes, axis=1).min(axis=1) > eps  # array[bool]
#     else:
#         return torch.stack(axes, dim=1).min(dim=1)[0] > eps  # Tensor[bool]


# def resize_boxes(
#     bbox: NdarrayOrTensor,
#     original_size: Union[Sequence, torch.Tensor, np.ndarray],
#     new_size: Union[Sequence, torch.Tensor, np.ndarray],
# ) -> NdarrayOrTensor:
#     """
#     modified from torchvision
#     Args:
#         bbox: source bounding box, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be STANDARD_MODE
#         original_size: source image size, Length of 2 or 3. Data format is list, or np.ndarray, or tensor of int
#         original_size: target image size, Length of 2 or 3. Data format is list, or np.ndarray, or tensor of int
#     """
#     if len(original_size) != len(new_size):
#         raise ValueError("The dimension of original image size should equal to the new image size")
#     spatial_dims = get_dimension(bbox, original_size)

#     original_size = convert_to_list(original_size)
#     new_size = convert_to_list(new_size)
#     zoom = [new_size[axis] / float(original_size[axis]) for axis in range(spatial_dims)]

#     return box_interp(bbox=bbox, zoom=zoom)
