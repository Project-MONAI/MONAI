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
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from copy import deepcopy
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


from monai.config.type_definitions import NdarrayOrTensor
from monai.data import box_utils
from monai.transforms.transform import Transform
from monai.utils import optional_import, convert_data_type
from monai.utils.enums import TransformBackends
from monai.utils.module import look_up_option

nib, _ = optional_import("nibabel")

__all__ = [
    "BoxConvertToStandard",
    "BoxConvertMode",
    "BoxClipToImage",
    "BoxFlip",
]

RandRange = Optional[Union[Sequence[Union[Tuple[float, float], float]], float]]


class BoxConvertMode(Transform):
    """
    Convert input boxes to standard mode
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, mode1: str, mode2: str) -> None:
        self.mode1 = mode1
        self.mode2 = mode2

    def __call__(self, bbox: NdarrayOrTensor) -> NdarrayOrTensor:
        # convert bbox to torch tensor
        if isinstance(bbox, np.ndarray):
            bbox_tensor = torch.from_numpy(bbox)
        else:
            bbox_tensor = bbox

        # clip box to the image and (optional) remove empty box
        bbox_standard = box_utils.box_convert_mode(bbox_tensor, mode1=self.mode1, mode2=self.mode2)

        if isinstance(bbox, np.ndarray):
            bbox_standard = bbox_standard.cpu().numpy()
        return bbox_standard


class BoxConvertToStandard(Transform):
    """
    Convert input boxes to standard mode
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, mode: str) -> None:
        self.converter = BoxConvertMode(mode1=mode, mode2=None)

    def __call__(self, bbox: NdarrayOrTensor) -> NdarrayOrTensor:
        # convert bbox to torch tensor
        return self.converter(bbox)

class BoxAffine(Transform):
    """
    Applys affine matrix to the bbox
    Args:
        invert_affine: whether to apply inversed affine matrix
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, mode: str, invert_affine: bool) -> None:
        self.mode = mode
        self.invert_affine = invert_affine
        

    def __call__(self, bbox: NdarrayOrTensor, affine: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            affine: affine matric to be applied to the box coordinate
        """
        # convert bbox to torch tensor
        if affine is None:
            return bbox
            
        if isinstance(bbox, np.ndarray):
            bbox_tensor = torch.from_numpy(bbox)
        else:
            bbox_tensor = bbox

        if isinstance(affine, np.ndarray):
            affine_tensor = torch.from_numpy(affine).to(bbox_tensor.dtype)
        else:
            affine_tensor = affine.to(bbox_tensor.dtype)

        if self.invert_affine:
            affine_tensor = torch.linalg.inv(affine_tensor)
        
        return box_utils.box_affine(bbox_tensor, affine=affine_tensor, mode=self.mode)


class BoxClipToImage(Transform):
    """
    Clip the bounding Boxes to makes sure they are within the image.

    Args:
        remove_empty: whether to remove the boxes that are actually empty
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        mode: str = None,
        remove_empty: bool = True,
    ) -> None:
        self.mode = mode
        self.remove_empty = remove_empty

    def __call__(
        self, bbox: NdarrayOrTensor, image_size: Union[Sequence[int], torch.Tensor, np.ndarray]
    ) -> NdarrayOrTensor:
        # convert bbox to torch tensor
        if isinstance(bbox, np.ndarray):
            bbox_tensor = torch.from_numpy(bbox)
        else:
            bbox_tensor = bbox

        # clip box to the image and (optional) remove empty box
        bbox_clip, _ = box_utils.box_clip_to_image(bbox_tensor, image_size, self.mode, self.remove_empty)

        if isinstance(bbox, np.ndarray):
            bbox_clip = bbox_clip.cpu().numpy()
        return bbox_clip


class BoxFlip(Transform):
    """
    Reverses the box coordinates along the given spatial axis. Preserves shape.
    We suggest performing BoxClipToImage before this transform.

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.

    """

    def __init__(self, spatial_axis: Optional[Union[Sequence[int], int]] = None, mode: str = None) -> None:
        self.spatial_axis = spatial_axis
        self.mode = mode

    def __call__(
        self, bbox: NdarrayOrTensor, image_size: Union[Sequence[int], torch.Tensor, np.ndarray]
    ) -> NdarrayOrTensor:
        # convert bbox to torch tensor
        if isinstance(bbox, np.ndarray):
            bbox_tensor = torch.from_numpy(bbox)
        else:
            bbox_tensor = bbox

        if self.spatial_axis is None:
            self.spatial_axis = list(range(0, len(image_size)))
        if isinstance(self.spatial_axis, int):
            self.spatial_axis = [self.spatial_axis]

        if self.mode is None:
            self.mode = box_utils.get_standard_mode(len(image_size) )
        self.mode = look_up_option(self.mode, supported=box_utils.STANDARD_MODE)
        spatial_dims = box_utils.get_dimension(mode=self.mode)

        # flip box
        flip_bbox_tensor = deepcopy(bbox_tensor)
        for axis in self.spatial_axis:
            flip_bbox_tensor[:, axis + spatial_dims] = image_size[axis] - bbox_tensor[:, axis] - box_utils.TO_REMOVE
            flip_bbox_tensor[:, axis] = image_size[axis] - bbox_tensor[:, axis + spatial_dims] - box_utils.TO_REMOVE

        if isinstance(bbox, np.ndarray):
            return flip_bbox_tensor.cpu().numpy()
        else:
            return flip_bbox_tensor


class BoxToBoxMask(Transform):
    """
    Convert box to int8 image, which has the same size with the input image,
    Each channel represents one box. The box region will have intensity of label, the background intensity is self.bg_label
    Box mask may take a lot of memory, so we generate box mask as numpy array

    Args:
        bg_label: background label for the output box image, just in case one of the fg label is 0
    """

    def __init__(self, mode: str = None, bg_label: int = -1 ) -> None:
        self.mode = mode
        self.bg_label = bg_label


    def __call__(
        self, bbox: NdarrayOrTensor, image_size: Union[Sequence[int], torch.Tensor, np.ndarray], label: Union[Sequence[int], torch.Tensor, np.ndarray]
    ) -> NdarrayOrTensor:
        if self.mode is None:
            self.mode = box_utils.get_standard_mode(len(image_size) )
        self.mode = look_up_option(self.mode, supported=box_utils.STANDARD_MODE)
        spatial_dims = box_utils.get_dimension(mode=self.mode)

        label = box_utils.convert_to_list(label)
        if self.bg_label >= min(label):
            raise ValueError(f"bg_label should be smaller than any foreground box label. min(box_label)={min(label)}, while bg_label={self.bg_label}")

        if len(label) != bbox.shape[0]:
            raise ValueError("Number of label should equal to number of bbox.")
        
        bbox_mask = np.ones([len(label)]+image_size,dtype=np.int8)*np.int8(self.bg_label)
        bbox,_,_ = convert_data_type(bbox,dtype=np.int16)
        for b in range(bbox.shape[0]):
            if spatial_dims == 2:
                bbox_mask[b, bbox[b,0]:bbox[b,spatial_dims], bbox[b,1]:bbox[b,1+spatial_dims] ] = np.int8(label[b])
            if spatial_dims == 3:
                bbox_mask[b, bbox[b,0]:bbox[b,spatial_dims], bbox[b,1]:bbox[b,1+spatial_dims], bbox[b,2]:bbox[b,2+spatial_dims] ] = np.int8(label[b])

        # if isinstance(bbox, torch.Tensor):
        #     bbox_mask = torch.from_numpy(bbox_mask)

        return bbox_mask

class BoxMaskToBox(Transform):
    """
    Convert binary image to box, which has the same size with the input image

    Args:
        bg_label: background label for the output box image
    """

    def __init__(self, mode: str = None, bg_label: int = -1 ) -> None:
        self.mode = mode
        self.bg_label = bg_label


    def __call__(self, bbox_mask: NdarrayOrTensor) -> NdarrayOrTensor:
        image_size = list(bbox_mask.shape[1:])
        if self.mode is None:
            self.mode = box_utils.get_standard_mode(len(image_size) )
        self.mode = look_up_option(self.mode, supported=box_utils.STANDARD_MODE)
        spatial_dims = box_utils.get_dimension(mode=self.mode)

        if isinstance(bbox_mask, torch.Tensor):
            bbox_mask = bbox_mask.cpu().detach().numpy()
        
        bbox = []
        label = []
        for b in range(bbox_mask.shape[0]):
            fg_indices = np.nonzero(bbox_mask[b,...]-self.bg_label)
            if fg_indices[0].shape[0] == 0:
                continue
            bbox_b = []
            for fd_i in fg_indices:
                bbox_b.append(min(fd_i)) # top left corner
            for fd_i in fg_indices:
                bbox_b.append(max(fd_i)+1) # bottom right corner
            if spatial_dims == 2:
                label.append(bbox_mask[b,bbox_b[0],bbox_b[1]])
            if spatial_dims == 3:
                label.append(bbox_mask[b,bbox_b[0],bbox_b[1],bbox_b[2]])
            bbox.append(bbox_b)

        return np.asarray(bbox),np.asarray(label)
