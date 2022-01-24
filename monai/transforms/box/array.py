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
from monai.utils import optional_import
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

        # flip box
        flip_bbox_tensor = deepcopy(bbox_tensor)
        for axis in self.spatial_axis:
            flip_bbox_tensor[:, 2 * axis + 1] = image_size[axis] - bbox_tensor[:, 2 * axis] - box_utils.TO_REMOVE
            flip_bbox_tensor[:, 2 * axis] = image_size[axis] - bbox_tensor[:, 2 * axis + 1] - box_utils.TO_REMOVE

        if isinstance(bbox, np.ndarray):
            return flip_bbox_tensor.cpu().numpy()
        else:
            return flip_bbox_tensor
