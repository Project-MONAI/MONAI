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
from typing import Optional, Sequence, Union

import torch

from monai.utils import look_up_option
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.box_utils import COMPUTE_DTYPE, TO_REMOVE, get_spatial_dims
from monai.transforms.utils import create_scale
from monai.utils.misc import ensure_tuple, ensure_tuple_rep
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type


def _apply_affine_to_points(points: torch.Tensor, affine: torch.Tensor, include_shift: bool = True) -> torch.Tensor:
    """
    This internal function applies affine matrices to the point coordinate

    Args:
        points: point coordinates, Nx2 or Nx3 torch tensor or ndarray, representing [x, y] or [x, y, z]
        affine: affine matrix to be applied to the point coordinates, sized (spatial_dims+1,spatial_dims+1)
        include_shift: default True, whether the function apply translation (shift) in the affine transform

    Returns:
        transformed point coordinates, with same data type as ``points``, does not share memory with ``points``
    """

    spatial_dims = get_spatial_dims(points=points)

    # compute new points
    if include_shift:
        # append 1 to form Nx(spatial_dims+1) vector, then transpose
        points_affine = torch.cat(
            [points, torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)], dim=1
        ).transpose(0, 1)
        # apply affine
        points_affine = torch.matmul(affine, points_affine)
        # remove appended 1 and transpose back
        points_affine = points_affine[:spatial_dims, :].transpose(0, 1)
    else:
        points_affine = points.transpose(0, 1)
        points_affine = torch.matmul(affine[:spatial_dims, :spatial_dims], points_affine)
        points_affine = points_affine.transpose(0, 1)

    return points_affine


def apply_affine_to_boxes(boxes: NdarrayOrTensor, affine: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    This function applies affine matrices to the boxes

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be StandardMode
        affine: affine matrix to be applied to the box coordinates, sized (spatial_dims+1,spatial_dims+1)

    Returns:
        returned affine transformed boxes, with same data type as ``boxes``, does not share memory with ``boxes``
    """

    # convert numpy to tensor if needed
    boxes_t, *_ = convert_data_type(boxes, torch.Tensor)

    # some operation does not support torch.float16
    # convert to float32

    boxes_t = boxes_t.to(dtype=COMPUTE_DTYPE)
    affine_t, *_ = convert_to_dst_type(src=affine, dst=boxes_t)

    spatial_dims = get_spatial_dims(boxes=boxes_t)

    # affine transform left top and bottom right points
    # might flipped, thus lt may not be left top any more
    lt: torch.Tensor = _apply_affine_to_points(boxes_t[:, :spatial_dims], affine_t, include_shift=True)
    rb: torch.Tensor = _apply_affine_to_points(boxes_t[:, spatial_dims:], affine_t, include_shift=True)

    # make sure lt_new is left top, and rb_new is bottom right
    lt_new, _ = torch.min(torch.stack([lt, rb], dim=2), dim=2)
    rb_new, _ = torch.max(torch.stack([lt, rb], dim=2), dim=2)

    boxes_t_affine = torch.cat([lt_new, rb_new], dim=1)

    # convert tensor back to numpy if needed
    boxes_affine: NdarrayOrTensor
    boxes_affine, *_ = convert_to_dst_type(src=boxes_t_affine, dst=boxes)
    return boxes_affine


def zoom_boxes(boxes: NdarrayOrTensor, zoom: Union[Sequence[float], float]) -> NdarrayOrTensor:
    """
    Zoom boxes

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be StandardMode
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.

    Returns:
        zoomed boxes, with same data type as ``boxes``, does not share memory with ``boxes``

    Example:
        .. code-block:: python

            boxes = torch.ones(1,4)
            zoom_boxes(boxes, zoom=[0.5,2.2]) #  will return tensor([[0.5, 2.2, 0.5, 2.2]])
    """
    spatial_dims = get_spatial_dims(boxes=boxes)

    # generate affine transform corresponding to ``zoom``
    affine = create_scale(spatial_dims=spatial_dims, scaling_factor=zoom)

    return apply_affine_to_boxes(boxes=boxes, affine=affine)


def resize_boxes(
    boxes: NdarrayOrTensor, src_spatial_size: Union[Sequence[int], int], dst_spatial_size: Union[Sequence[int], int]
) -> NdarrayOrTensor:
    """
    Resize boxes when the corresponding image is resized

    Args:
        boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        src_spatial_size: source image spatial size.
        dst_spatial_size: target image spatial size.

    Returns:
        resized boxes, with same data type as ``boxes``, does not share memory with ``boxes``

    Example:
        .. code-block:: python

            boxes = torch.ones(1,4)
            src_spatial_size = [100, 100]
            dst_spatial_size = [128, 256]
            resize_boxes(boxes, src_spatial_size, dst_spatial_size) #  will return tensor([[1.28, 2.56, 1.28, 2.56]])
    """
    spatial_dims: int = get_spatial_dims(boxes=boxes)

    src_spatial_size = ensure_tuple_rep(src_spatial_size, spatial_dims)
    dst_spatial_size = ensure_tuple_rep(dst_spatial_size, spatial_dims)

    zoom = [dst_spatial_size[axis] / float(src_spatial_size[axis]) for axis in range(spatial_dims)]

    return zoom_boxes(boxes=boxes, zoom=zoom)


def flip_boxes(
    boxes: NdarrayOrTensor,
    spatial_size: Union[Sequence[int], int],
    flip_axes: Optional[Union[Sequence[int], int]] = None,
) -> NdarrayOrTensor:
    """
    Flip boxes when the corresponding image is flipped

    Args:
        boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
        spatial_size: image spatial size.
        flip_axes: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.

    Returns:
        flipped boxes, with same data type as ``boxes``, does not share memory with ``boxes``
    """
    spatial_dims: int = get_spatial_dims(boxes=boxes)
    spatial_size = ensure_tuple_rep(spatial_size, spatial_dims)
    if flip_axes is None:
        flip_axes = tuple(range(0, spatial_dims))
    flip_axes = ensure_tuple(flip_axes)

    # flip box
    flip_boxes = deepcopy(boxes)
    for axis in flip_axes:
        flip_boxes[:, axis + spatial_dims] = spatial_size[axis] - boxes[:, axis] - TO_REMOVE
        flip_boxes[:, axis] = spatial_size[axis] - boxes[:, axis + spatial_dims] - TO_REMOVE

    return flip_boxes

def convert_boxes_to_mask(
    boxes: NdarrayOrTensor, 
    spatial_size: Union[Sequence[int], int],
    labels: Union[Sequence[int], torch.Tensor, np.ndarray],
    bg_label: int = -1, 
    ellipse_mask: bool = False
    ) -> NdarrayOrTensor:
    """
    Convert box to int8 image, which has the same size with the input image,
    Each channel represents one box. The box region will have intensity of labels, the background intensity is bg_label
    Box mask may take a lot of memory, so we generate box mask as numpy array

    Args:
        bg_label: background labels for the output box image, just in case one of the fg labels is 0
    """
    spatial_dims: int = get_spatial_dims(boxes=boxes)
    spatial_size = ensure_tuple_rep(spatial_size, spatial_dims)

    labels = box_utils.convert_to_list(labels)
    # if no box, return empty mask
    if len(labels)==0:
        return np.ones([1]+spatial_size,dtype=np.int8)*np.int8(bg_label)
    
    if bg_label >= min(labels):
        raise ValueError(f"bg_label should be smaller than any foreground box labels. min(box_labels)={min(labels)}, while bg_label={bg_label}")

    if len(labels) != boxes.shape[0]:
        raise ValueError("Number of labels should equal to number of boxes.")
    
    boxes_mask = np.ones([len(labels)]+spatial_size,dtype=np.int8)*np.int8(bg_label)
    boxes,_,_ = convert_data_type(boxes,dtype=np.int16)
    for b in range(boxes.shape[0]):
        # draw a circle/ball mask
        box_size = [boxes[b,axis+spatial_dims].item()-boxes[b,axis].item() for axis in range(spatial_dims)]
        if self.ellipse_mask:
            max_box_size = max(box_size)
            radius = max_box_size/2.0
            center = (max_box_size-1)/2.0
            boxes_only_mask = np.ones([max_box_size]*spatial_dims,dtype=np.int8)*np.int8(bg_label) # a square/cube mask
            if spatial_dims == 2:
                Y, X = np.ogrid[:max_box_size, :max_box_size]
                dist_from_center = (X-center)**2 + (Y-center)**2
            elif spatial_dims == 3:
                Y, X, Z = np.ogrid[:max_box_size, :max_box_size, :max_box_size]
                dist_from_center = (X-center)**2 + (Y-center)**2+ (Z-center)**2
            boxes_only_mask[dist_from_center <= radius**2] = np.int8(labels[b])

            # squeeze it to a ellipse/ellipsoid
            zoom_factor = [box_size[axis]/float(max_box_size) for axis in range(spatial_dims)]
            boxes_only_mask = scipy.ndimage.zoom(boxes_only_mask,zoom=zoom_factor,mode='nearest',prefilter=False)
        else:
            boxes_only_mask = np.ones(box_size,dtype=np.int8)*np.int8(labels[b]) 

        # apply to global mask
        if spatial_dims == 2:
            boxes_mask[b, boxes[b,0]:boxes[b,spatial_dims], boxes[b,1]:boxes[b,1+spatial_dims] ] = boxes_only_mask
        if spatial_dims == 3:
            boxes_mask[b, boxes[b,0]:boxes[b,spatial_dims], boxes[b,1]:boxes[b,1+spatial_dims], boxes[b,2]:boxes[b,2+spatial_dims] ] = boxes_only_mask

    # if isinstance(boxes, torch.Tensor):
    #     boxes_mask = torch.from_numpy(boxes_mask)

    return boxes_mask

def convert_mask_to_boxes(boxes_mask: NdarrayOrTensor, bg_label: int = -1) -> NdarrayOrTensor:
    """
    Convert binary mask image to box, which has the same size with the input image

    Args:
        boxes_mask: sized (num_box, H, W) or (num_box, H, W, D)
        bg_label: background labels for the boxes_mask

    Return:
        - boxes
        - labelss
    """
    look_up_option(len(boxes_mask.shape), [3,4])
    spatial_size = list(boxes_mask.shape[1:])
    spatial_dims = box_utils.get_spatial_dims(spatial_size=spatial_size)

    if isinstance(boxes_mask, torch.Tensor):
        boxes_mask = boxes_mask.cpu().detach().numpy()
    
    boxes = []
    labels = []
    for b in range(boxes_mask.shape[0]):
        fg_indices = np.nonzero(boxes_mask[b,...]-bg_label)
        if fg_indices[0].shape[0] == 0:
            continue
        boxes_b = []
        for fd_i in fg_indices:
            boxes_b.append(min(fd_i)) # top left corner
        for fd_i in fg_indices:
            boxes_b.append(max(fd_i)+1) # bottom right corner
        if spatial_dims == 2:
            labels.append(boxes_mask[b,boxes_b[0],boxes_b[1]])
        if spatial_dims == 3:
            labels.append(boxes_mask[b,boxes_b[0],boxes_b[1],boxes_b[2]])
        boxes.append(boxes_b)

    if len(boxes) == 0:
        return np.zeros([0,2*spatial_dims]), np.zeros([0])
    return np.asarray(boxes),np.asarray(labels)
