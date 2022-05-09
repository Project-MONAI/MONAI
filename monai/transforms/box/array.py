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
from itertools import chain

import numpy as np
import torch
import scipy


from monai.config.type_definitions import NdarrayOrTensor
from monai.data import box_utils
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
from monai.transforms.utils_pytorch_numpy_unification import floor_divide, maximum
from monai.utils import (
    optional_import,
    convert_data_type,
    ensure_tuple_rep,
    NumpyPadMode,
    PytorchPadMode,
)
from monai.transforms.utils import (
    compute_divisible_spatial_size,
)
from monai.transforms.croppad.array import (
    SpatialCrop,
    BorderPad
)

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
            affine_tensor = torch.inverse(affine_tensor)

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
        self, bbox: NdarrayOrTensor, label: NdarrayOrTensor, image_size: Union[Sequence[int], torch.Tensor, np.ndarray]
    ) -> NdarrayOrTensor:
        # convert bbox to torch tensor
        if isinstance(bbox, np.ndarray):
            bbox_tensor = torch.from_numpy(bbox)
        else:
            bbox_tensor = bbox

        # clip box to the image and (optional) remove empty box
        bbox_clip, keep = box_utils.box_clip_to_image(bbox_tensor, image_size, self.mode, self.remove_empty)

        if isinstance(bbox, np.ndarray):
            bbox_clip = bbox_clip.cpu().numpy()
        return bbox_clip, label[keep]


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

    def __init__(self, mode: str = None, bg_label: int = -1, ellipse_mask: bool = False) -> None:
        self.mode = mode
        self.bg_label = bg_label
        self.ellipse_mask = ellipse_mask


    def __call__(
        self, bbox: NdarrayOrTensor, image_size: Union[Sequence[int], torch.Tensor, np.ndarray], label: Union[Sequence[int], torch.Tensor, np.ndarray]
    ) -> NdarrayOrTensor:
        if self.mode is None:
            self.mode = box_utils.get_standard_mode(len(image_size) )
        self.mode = look_up_option(self.mode, supported=box_utils.STANDARD_MODE)
        spatial_dims = box_utils.get_dimension(mode=self.mode)

        label = box_utils.convert_to_list(label)
        # if no box, return empty mask
        if len(label)==0:
            return np.ones([1]+image_size,dtype=np.int8)*np.int8(self.bg_label)

        if self.bg_label >= min(label):
            raise ValueError(f"bg_label should be smaller than any foreground box label. min(box_label)={min(label)}, while bg_label={self.bg_label}")

        if len(label) != bbox.shape[0]:
            raise ValueError("Number of label should equal to number of bbox.")

        bbox_mask = np.ones([len(label)]+image_size,dtype=np.int8)*np.int8(self.bg_label)
        bbox,_,_ = convert_data_type(bbox,dtype=np.int16)
        for b in range(bbox.shape[0]):
            # draw a circle/ball mask
            box_size = [bbox[b,axis+spatial_dims].item()-bbox[b,axis].item() for axis in range(spatial_dims)]
            if self.ellipse_mask:
                max_box_size = max(box_size)
                radius = max_box_size/2.0
                center = (max_box_size-1)/2.0
                bbox_only_mask = np.ones([max_box_size]*spatial_dims,dtype=np.int8)*np.int8(self.bg_label) # a square/cube mask
                if spatial_dims == 2:
                    Y, X = np.ogrid[:max_box_size, :max_box_size]
                    dist_from_center = (X-center)**2 + (Y-center)**2
                elif spatial_dims == 3:
                    Y, X, Z = np.ogrid[:max_box_size, :max_box_size, :max_box_size]
                    dist_from_center = (X-center)**2 + (Y-center)**2+ (Z-center)**2
                bbox_only_mask[dist_from_center <= radius**2] = np.int8(label[b])

                # squeeze it to a ellipse/ellipsoid
                zoom_factor = [box_size[axis]/float(max_box_size) for axis in range(spatial_dims)]
                bbox_only_mask = scipy.ndimage.zoom(bbox_only_mask,zoom=zoom_factor,mode='nearest',prefilter=False)
            else:
                bbox_only_mask = np.ones(box_size,dtype=np.int8)*np.int8(label[b])

            # apply to global mask
            if spatial_dims == 2:
                bbox_mask[b, bbox[b,0]:bbox[b,spatial_dims], bbox[b,1]:bbox[b,1+spatial_dims] ] = bbox_only_mask
            if spatial_dims == 3:
                bbox_mask[b, bbox[b,0]:bbox[b,spatial_dims], bbox[b,1]:bbox[b,1+spatial_dims], bbox[b,2]:bbox[b,2+spatial_dims] ] = bbox_only_mask

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

        if len(bbox) == 0:
            return np.zeros([0,2*spatial_dims]), np.zeros([0])
        return np.asarray(bbox),np.asarray(label)

class BoxSpatialCropPad(Transform):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial (channel-first) data.
    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (allows for use of -ve indexing and `None`)
        - a spatial center and size
        - the start and end coordinates of the ROI
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        roi_center: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_size: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_start: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_end: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_slices: Optional[Sequence[slice]] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
        """
        roi_start_torch: torch.Tensor

        if roi_slices:
            if not all(s.step is None or s.step == 1 for s in roi_slices):
                raise ValueError("Only slice steps of 1/None are currently supported")
            self.slices = list(roi_slices)
        else:
            if roi_center is not None and roi_size is not None:
                roi_center, *_ = convert_data_type(
                    data=roi_center, output_type=torch.Tensor, dtype=torch.int16, wrap_sequence=True
                )
                roi_size, *_ = convert_to_dst_type(src=roi_size, dst=roi_center, wrap_sequence=True)
                _zeros = torch.zeros_like(roi_center)
                roi_start_torch = roi_center - floor_divide(roi_size, 2)  # type: ignore
                roi_start_torch = maximum(roi_start_torch, _zeros)  # type: ignore
                roi_end_torch = roi_start_torch + roi_size
                roi_end_torch = maximum(roi_end_torch, roi_start_torch)
            else:
                if roi_start is None or roi_end is None:
                    raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
                roi_start_torch, *_ = convert_data_type(
                    data=roi_start, output_type=torch.Tensor, dtype=torch.int16, wrap_sequence=True
                )
                roi_end_torch, *_ = convert_to_dst_type(src=roi_end, dst=roi_start_torch, wrap_sequence=True)
                roi_start_torch = maximum(roi_start_torch, torch.zeros_like(roi_start_torch))  # type: ignore
                roi_end_torch = maximum(roi_end_torch, roi_start_torch)
            # convert to slices (accounting for 1d)
            if roi_start_torch.numel() == 1:
                self.slices = [slice(int(roi_start_torch.item()), int(roi_end_torch.item()))]
            else:
                self.slices = [slice(int(s), int(e)) for s, e in zip(roi_start_torch.tolist(), roi_end_torch.tolist())]

    def __call__(self, bbox: NdarrayOrTensor, label: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        # print(self.slices)
        spatial_dims = min(len(self.slices), bbox.shape[1]//2)  # spatial dims
        patch_box = [0]*6
        for axis in range(spatial_dims):
            patch_box[axis] = self.slices[axis].start
            patch_box[axis+spatial_dims] = self.slices[axis].stop
        new_bbox, keep = box_utils.box_clip_to_patch(bbox,patch_box)
        new_label = label[keep]
        return new_bbox, new_label

class BoxCropForeground(Transform):
    """
    Crop an image using a bounding box. The bounding box is generated by selecting foreground using select_fn
    at channels channel_indices. margin is added in each spatial dimension of the bounding box.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box of foreground object.
    For example:
    .. code-block:: python
        image = np.array(
            [[[0, 0, 0, 0, 0],
              [0, 1, 2, 1, 0],
              [0, 1, 3, 2, 0],
              [0, 1, 2, 1, 0],
              [0, 0, 0, 0, 0]]])  # 1x5x5, single channel 5x5 image
        def threshold_at_one(x):
            # threshold at 1
            return x > 1
        cropper = CropForeground(select_fn=threshold_at_one, margin=0)
        print(cropper(image))
        [[[2, 1],
          [3, 2],
          [2, 1]]]
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        margin: Union[Sequence[int], int] = 0,
        allow_smaller: bool = True,
        return_coords: bool = False,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = NumpyPadMode.CONSTANT,
        **np_kwargs,
    ) -> None:
        """
        Args:
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            allow_smaller: when computing box size with `margin`, whether allow the image size to be smaller
                than box size, default to `True`. if the margined size is bigger than image size, will pad with
                specified `mode`.
            return_coords: whether return the coordinates of spatial bounding box for foreground.
            k_divisible: make each spatial dimension to be divisible by k, default to 1.
                if `k_divisible` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            np_kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
                more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        """
        self.margin = margin
        self.allow_smaller = allow_smaller
        self.return_coords = return_coords
        self.k_divisible = k_divisible
        self.mode: NumpyPadMode = look_up_option(mode, NumpyPadMode)
        self.np_kwargs = np_kwargs
        self.allow_smaller = allow_smaller

    def compute_bounding_box(self, bbox: NdarrayOrTensor, image_size: NdarrayOrTensor,):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.
        """
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().detach().numpy()
        spatial_dims = bbox.shape[1]//2
        self.margin = ensure_tuple_rep(self.margin, spatial_dims)

        box_start = np.amin(bbox[:,:spatial_dims],axis=0)
        box_end = np.amax(bbox[:,spatial_dims:],axis=0)

        for axis in range(spatial_dims):
            box_start[axis] -= self.margin[axis]
            box_end[axis] += self.margin[axis]
            if self.allow_smaller:
                box_start[axis] = max(box_start[axis], 0)
                box_end[axis] = min(box_end[axis], image_size[axis])

        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        orig_spatial_size = box_end_ - box_start_
        # make the spatial size divisible by `k`
        spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=self.k_divisible))
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size

        return box_start_, box_end_

    def crop_pad(
        self,
        img: NdarrayOrTensor,
        box_start: np.ndarray,
        box_end: np.ndarray,
        mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
    ):
        """
        Crop and pad based on the bounding box.
        """
        cropped = SpatialCrop(roi_start=box_start, roi_end=box_end)(img)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(box_end - np.asarray(img.shape[1:]), 0)
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        return BorderPad(spatial_border=pad, mode=mode or self.mode, **self.np_kwargs)(cropped)

    def __call__(self, img: NdarrayOrTensor,bbox: NdarrayOrTensor,label: NdarrayOrTensor, mode: Optional[Union[NumpyPadMode, str]] = None):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        image_size = img.shape[1:]
        box_start, box_end = self.compute_bounding_box(bbox,image_size)
        cropped = self.crop_pad(img, box_start, box_end, mode)

        boxcropper = BoxSpatialCropPad(roi_start=box_start, roi_end=box_end)
        bbox, label = boxcropper(bbox,label)

        if self.return_coords:
            return cropped, box_start, box_end
        return cropped

class BoxZoom(Transform):
    """
    Zooms an ND Box with same padding or slicing setting with Zoom().

    Args:
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        keep_size: Should keep original size (padding/slicing if needed), default is True.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        zoom: Union[Sequence[float], float],
        keep_size: bool = True,
        **kwargs,
    ) -> None:
        self.zoom = zoom
        self.keep_size = keep_size
        self.kwargs = kwargs

    def __call__(
        self,
        bbox: NdarrayOrTensor,
        orig_image_size: Union[Sequence[int], torch.Tensor, np.ndarray] = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            bbox: Nx 4 or Nx6
            orig_image_size: original image size before zooming
        """
        spatial_dims = bbox.shape[1]//2
        _zoom = ensure_tuple_rep(self.zoom, spatial_dims)  # match the spatial image dim

        for axis in range(spatial_dims):
            bbox[:,axis] = bbox[:,axis] * _zoom[axis]
            bbox[:,axis+spatial_dims] = bbox[:,axis+spatial_dims] * _zoom[axis]

        zoomed_size = np.asarray([int(round(orig_image_size[axis]*_zoom[axis])) for axis in range(spatial_dims)])
        if self.keep_size and not np.allclose(orig_image_size, zoomed_size):
            for axis, (od, zd) in enumerate(zip(orig_image_size, zoomed_size)):
                diff = od - zd
                half = abs(diff) // 2
                if diff > 0:  # need padding (half, diff - half)
                    bbox[:,axis] = bbox[:,axis] + half
                    bbox[:,axis+spatial_dims] = bbox[:,axis+spatial_dims] + half
                elif diff < 0:  # need slicing (half, half + od)
                    bbox[:,axis] = bbox[:,axis] - half
                    bbox[:,axis+spatial_dims] = bbox[:,axis+spatial_dims] - half
        return bbox
