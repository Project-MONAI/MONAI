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
A collection of "vanilla" transforms for box operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from monai.config.type_definitions import DtypeLike, NdarrayOrTensor, NdarrayTensor
from monai.data.box_utils import (
    BoxMode,
    clip_boxes_to_image,
    convert_box_mode,
    convert_box_to_standard_mode,
    get_spatial_dims,
    spatial_crop_boxes,
    standardize_empty_box,
)
from monai.transforms import Rotate90, SpatialCrop
from monai.transforms.transform import Transform
from monai.utils import ensure_tuple, ensure_tuple_rep, fall_back_tuple, look_up_option
from monai.utils.enums import TransformBackends

from .box_ops import (
    apply_affine_to_boxes,
    convert_box_to_mask,
    convert_mask_to_box,
    flip_boxes,
    resize_boxes,
    rot90_boxes,
    select_labels,
    zoom_boxes,
)

__all__ = [
    "StandardizeEmptyBox",
    "ConvertBoxToStandardMode",
    "ConvertBoxMode",
    "AffineBox",
    "ZoomBox",
    "ResizeBox",
    "FlipBox",
    "ClipBoxToImage",
    "BoxToMask",
    "MaskToBox",
    "SpatialCropBox",
    "RotateBox90",
]


class StandardizeEmptyBox(Transform):
    """
    When boxes are empty, this transform standardize it to shape of (0,4) or (0,6).

    Args:
        spatial_dims: number of spatial dimensions of the bounding boxes.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, spatial_dims: int) -> None:
        self.spatial_dims = spatial_dims

    def __call__(self, boxes: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            boxes: source bounding boxes, Nx4 or Nx6 or 0xM torch tensor or ndarray.
        """
        return standardize_empty_box(boxes, spatial_dims=self.spatial_dims)


class ConvertBoxMode(Transform):
    """
    This transform converts the boxes in src_mode to the dst_mode.

    Args:
        src_mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
        dst_mode: target box mode. If it is not given, this func will assume it is ``StandardMode()``.

    Note:
        ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
        also represented as "xyxy" for 2D and "xyzxyz" for 3D.

        src_mode and dst_mode can be:
            #. str: choose from :class:`~monai.utils.enums.BoxModeName`, for example,
                - "xyxy": boxes has format [xmin, ymin, xmax, ymax]
                - "xyzxyz": boxes has format [xmin, ymin, zmin, xmax, ymax, zmax]
                - "xxyy": boxes has format [xmin, xmax, ymin, ymax]
                - "xxyyzz": boxes has format [xmin, xmax, ymin, ymax, zmin, zmax]
                - "xyxyzz": boxes has format [xmin, ymin, xmax, ymax, zmin, zmax]
                - "xywh": boxes has format [xmin, ymin, xsize, ysize]
                - "xyzwhd": boxes has format [xmin, ymin, zmin, xsize, ysize, zsize]
                - "ccwh": boxes has format [xcenter, ycenter, xsize, ysize]
                - "cccwhd": boxes has format [xcenter, ycenter, zcenter, xsize, ysize, zsize]
            #. BoxMode class: choose from the subclasses of :class:`~monai.data.box_utils.BoxMode`, for example,
                - CornerCornerModeTypeA: equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB: equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC: equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode: equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode: equivalent to "ccwh" or "cccwhd"
            #. BoxMode object: choose from the subclasses of :class:`~monai.data.box_utils.BoxMode`, for example,
                - CornerCornerModeTypeA(): equivalent to "xyxy" or "xyzxyz"
                - CornerCornerModeTypeB(): equivalent to "xxyy" or "xxyyzz"
                - CornerCornerModeTypeC(): equivalent to "xyxy" or "xyxyzz"
                - CornerSizeMode(): equivalent to "xywh" or "xyzwhd"
                - CenterSizeMode(): equivalent to "ccwh" or "cccwhd"
            #. None: will assume mode is ``StandardMode()``

    Example:
        .. code-block:: python

            boxes = torch.ones(10,4)
            # convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
            box_converter = ConvertBoxMode(src_mode="xyxy", dst_mode="ccwh")
            box_converter(boxes)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        src_mode: str | BoxMode | type[BoxMode] | None = None,
        dst_mode: str | BoxMode | type[BoxMode] | None = None,
    ) -> None:
        self.src_mode = src_mode
        self.dst_mode = dst_mode

    def __call__(self, boxes: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Converts the boxes in src_mode to the dst_mode.

        Args:
            boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

        Returns:
            bounding boxes with target mode, with same data type as ``boxes``, does not share memory with ``boxes``
        """
        return convert_box_mode(boxes, src_mode=self.src_mode, dst_mode=self.dst_mode)


class ConvertBoxToStandardMode(Transform):
    """
    Convert given boxes to standard mode.
    Standard mode is "xyxy" or "xyzxyz",
    representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

    Args:
        mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
            It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.ConvertBoxMode` .

    Example:
        .. code-block:: python

            boxes = torch.ones(10,6)
            # convert boxes with format [xmin, xmax, ymin, ymax, zmin, zmax] to [xmin, ymin, zmin, xmax, ymax, zmax]
            box_converter = ConvertBoxToStandardMode(mode="xxyyzz")
            box_converter(boxes)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, mode: str | BoxMode | type[BoxMode] | None = None) -> None:
        self.mode = mode

    def __call__(self, boxes: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Convert given boxes to standard mode.
        Standard mode is "xyxy" or "xyzxyz",
        representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

        Args:
            boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``

        Returns:
            bounding boxes with standard mode, with same data type as ``boxes``, does not share memory with ``boxes``
        """
        return convert_box_to_standard_mode(boxes, mode=self.mode)


class AffineBox(Transform):
    """
    Applies affine matrix to the boxes
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, boxes: NdarrayOrTensor, affine: NdarrayOrTensor | None) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            affine: affine matrix to be applied to the box coordinate
        """
        if affine is None:
            return boxes

        return apply_affine_to_boxes(boxes, affine=affine)


class ZoomBox(Transform):
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

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, zoom: Sequence[float] | float, keep_size: bool = False, **kwargs: Any) -> None:
        self.zoom = zoom
        self.keep_size = keep_size
        self.kwargs = kwargs

    def __call__(self, boxes: NdarrayTensor, src_spatial_size: Sequence[int] | int | None = None) -> NdarrayTensor:
        """
        Args:
            boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            src_spatial_size: original image spatial size before zooming, used only when keep_size=True.
        """
        spatial_dims: int = get_spatial_dims(boxes=boxes)
        self._zoom = ensure_tuple_rep(self.zoom, spatial_dims)  # match the spatial image dim

        if not self.keep_size:
            return zoom_boxes(boxes, self._zoom)

        if src_spatial_size is None:
            raise ValueError("keep_size=True, src_spatial_size must be provided.")

        src_spatial_size = ensure_tuple_rep(src_spatial_size, spatial_dims)
        dst_spatial_size = [int(round(z * ss)) for z, ss in zip(self._zoom, src_spatial_size)]
        self._zoom = tuple(ds / float(ss) for ss, ds in zip(src_spatial_size, dst_spatial_size))
        zoomed_boxes = zoom_boxes(boxes, self._zoom)

        # See also keep_size in monai.transforms.spatial.array.Zoom()
        if not np.allclose(np.array(src_spatial_size), np.array(dst_spatial_size)):
            for axis, (od, zd) in enumerate(zip(src_spatial_size, dst_spatial_size)):
                diff = od - zd
                half = abs(diff) // 2
                if diff > 0:  # need padding (half, diff - half)
                    zoomed_boxes[:, axis] = zoomed_boxes[:, axis] + half
                    zoomed_boxes[:, axis + spatial_dims] = zoomed_boxes[:, axis + spatial_dims] + half
                elif diff < 0:  # need slicing (half, half + od)
                    zoomed_boxes[:, axis] = zoomed_boxes[:, axis] - half
                    zoomed_boxes[:, axis + spatial_dims] = zoomed_boxes[:, axis + spatial_dims] - half
        return zoomed_boxes


class ResizeBox(Transform):
    """
    Resize the input boxes when the corresponding image is
    resized to given spatial size (with scaling, not cropping/padding).

    Args:
        spatial_size: expected shape of spatial dimensions after resize operation.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        size_mode: should be "all" or "longest", if "all", will use `spatial_size` for all the spatial dims,
            if "longest", rescale the image so that only the longest side is equal to specified `spatial_size`,
            which must be an int number in this case, keeping the aspect ratio of the initial image, refer to:
            https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
            #albumentations.augmentations.geometric.resize.LongestMaxSize.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, spatial_size: Sequence[int] | int, size_mode: str = "all", **kwargs: Any) -> None:
        self.size_mode = look_up_option(size_mode, ["all", "longest"])
        self.spatial_size = spatial_size

    def __call__(self, boxes: NdarrayOrTensor, src_spatial_size: Sequence[int] | int) -> NdarrayOrTensor:  # type: ignore[override]
        """
        Args:
            boxes: source bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            src_spatial_size: original image spatial size before resizing.

        Raises:
            ValueError: When ``self.spatial_size`` length is less than ``boxes`` spatial dimensions.
        """
        input_ndim = get_spatial_dims(boxes=boxes)  # spatial ndim
        src_spatial_size_ = ensure_tuple_rep(src_spatial_size, input_ndim)

        if self.size_mode == "all":
            # spatial_size must be a Sequence if size_mode is 'all'
            output_ndim = len(ensure_tuple(self.spatial_size))
            if output_ndim != input_ndim:
                raise ValueError(
                    "len(spatial_size) must be greater or equal to img spatial dimensions, "
                    f"got spatial_size={output_ndim} img={input_ndim}."
                )
            spatial_size_ = fall_back_tuple(self.spatial_size, src_spatial_size_)
        else:  # for the "longest" mode
            if not isinstance(self.spatial_size, int):
                raise ValueError("spatial_size must be an int number if size_mode is 'longest'.")
            scale = self.spatial_size / max(src_spatial_size_)
            spatial_size_ = tuple(int(round(s * scale)) for s in src_spatial_size_)

        return resize_boxes(boxes, src_spatial_size_, spatial_size_)


class FlipBox(Transform):
    """
    Reverses the box coordinates along the given spatial axis. Preserves shape.

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, spatial_axis: Sequence[int] | int | None = None) -> None:
        self.spatial_axis = spatial_axis

    def __call__(self, boxes: NdarrayOrTensor, spatial_size: Sequence[int] | int):  # type: ignore
        """
        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            spatial_size: image spatial size.
        """

        return flip_boxes(boxes, spatial_size=spatial_size, flip_axes=self.spatial_axis)


class ClipBoxToImage(Transform):
    """
    Clip the bounding boxes and the associated labels/scores to make sure they are within the image.
    There might be multiple arrays of labels/scores associated with one array of boxes.

    Args:
        remove_empty: whether to remove the boxes and corresponding labels that are actually empty
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, remove_empty: bool = False) -> None:
        self.remove_empty = remove_empty

    def __call__(  # type: ignore
        self,
        boxes: NdarrayOrTensor,
        labels: Sequence[NdarrayOrTensor] | NdarrayOrTensor,
        spatial_size: Sequence[int] | int,
    ) -> tuple[NdarrayOrTensor, tuple | NdarrayOrTensor]:
        """
        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            labels: Sequence of array. Each element represents classification labels or scores
                corresponding to ``boxes``, sized (N,).
            spatial_size: The spatial size of the image where the boxes are attached. len(spatial_size) should be in [2, 3].

        Returns:
            - clipped boxes, does not share memory with original boxes
            - clipped labels, does not share memory with original labels

        Example:
            .. code-block:: python

                box_clipper = ClipBoxToImage(remove_empty=True)
                boxes = torch.ones(2, 6)
                class_labels = torch.Tensor([0, 1])
                pred_scores = torch.Tensor([[0.4,0.3,0.3], [0.5,0.1,0.4]])
                labels = (class_labels, pred_scores)
                spatial_size = [32, 32, 32]
                boxes_clip, labels_clip_tuple = box_clipper(boxes, labels, spatial_size)
        """
        spatial_dims: int = get_spatial_dims(boxes=boxes)
        spatial_size = ensure_tuple_rep(spatial_size, spatial_dims)  # match the spatial image dim

        boxes_clip, keep = clip_boxes_to_image(boxes, spatial_size, self.remove_empty)
        return boxes_clip, select_labels(labels, keep)


class BoxToMask(Transform):
    """
    Convert box to int16 mask image, which has the same size with the input image.

    Args:
        bg_label: background labels for the output mask image, make sure it is smaller than any foreground(fg) labels.
        ellipse_mask: bool.

            - If True, it assumes the object shape is close to ellipse or ellipsoid.
            - If False, it assumes the object shape is close to rectangle or cube and well occupies the bounding box.
            - If the users are going to apply random rotation as data augmentation, we suggest setting ellipse_mask=True
              See also Kalra et al. "Towards Rotation Invariance in Object Detection", ICCV 2021.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, bg_label: int = -1, ellipse_mask: bool = False) -> None:
        self.bg_label = bg_label
        self.ellipse_mask = ellipse_mask

    def __call__(  # type: ignore
        self, boxes: NdarrayOrTensor, labels: NdarrayOrTensor, spatial_size: Sequence[int] | int
    ) -> NdarrayOrTensor:
        """
        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``.
            labels: classification foreground(fg) labels corresponding to `boxes`, dtype should be int, sized (N,).
            spatial_size: image spatial size.

        Return:
            - int16 array, sized (num_box, H, W). Each channel represents a box.
                The foreground region in channel c has intensity of labels[c].
                The background intensity is bg_label.
        """
        return convert_box_to_mask(boxes, labels, spatial_size, self.bg_label, self.ellipse_mask)


class MaskToBox(Transform):
    """
    Convert int16 mask image to box, which has the same size with the input image.
    Pairs with :py:class:`monai.apps.detection.transforms.array.BoxToMask`.
    Please make sure the same ``min_fg_label`` is used when using the two transforms in pairs.

    Args:
        bg_label: background labels for the output mask image, make sure it is smaller than any foreground(fg) labels.
        box_dtype: output dtype for boxes
        label_dtype: output dtype for labels
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        bg_label: int = -1,
        box_dtype: DtypeLike | torch.dtype = torch.float32,
        label_dtype: DtypeLike | torch.dtype = torch.long,
    ) -> None:
        self.bg_label = bg_label
        self.box_dtype = box_dtype
        self.label_dtype = label_dtype

    def __call__(self, boxes_mask: NdarrayOrTensor) -> tuple[NdarrayOrTensor, NdarrayOrTensor]:
        """
        Args:
            boxes_mask: int16 array, sized (num_box, H, W). Each channel represents a box.
                The foreground region in channel c has intensity of labels[c].
                The background intensity is bg_label.

        Return:
            - bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``.
            - classification foreground(fg) labels, dtype should be int, sized (N,).
        """
        return convert_mask_to_box(boxes_mask, self.bg_label, self.box_dtype, self.label_dtype)


class SpatialCropBox(SpatialCrop):
    """
    General purpose box cropper when the corresponding image is cropped by SpatialCrop(*) with the same ROI.
    The difference is that we do not support negative indexing for roi_slices.

    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial boxes.

    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (do not allow for use of negative indexing)
        - a spatial center and size
        - the start and end coordinates of the ROI

    Args:
        roi_center: voxel coordinates for center of the crop ROI.
        roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
            will not crop that dimension of the image.
        roi_start: voxel coordinates for start of the crop ROI.
        roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
            use the end coordinate of image.
        roi_slices: list of slices for each of the spatial dimensions.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        roi_center: Sequence[int] | NdarrayOrTensor | None = None,
        roi_size: Sequence[int] | NdarrayOrTensor | None = None,
        roi_start: Sequence[int] | NdarrayOrTensor | None = None,
        roi_end: Sequence[int] | NdarrayOrTensor | None = None,
        roi_slices: Sequence[slice] | None = None,
    ) -> None:
        super().__init__(roi_center, roi_size, roi_start, roi_end, roi_slices)
        for s in self.slices:
            if s.start < 0 or s.stop < 0 or (s.step is not None and s.step < 0):
                raise ValueError("Currently negative indexing is not supported for SpatialCropBox.")

    def __call__(  # type: ignore[override]
        self, boxes: NdarrayTensor, labels: Sequence[NdarrayOrTensor] | NdarrayOrTensor
    ) -> tuple[NdarrayTensor, tuple | NdarrayOrTensor]:
        """
        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            labels: Sequence of array. Each element represents classification labels or scores

        Returns:
            - cropped boxes, does not share memory with original boxes
            - cropped labels, does not share memory with original labels

        Example:
            .. code-block:: python

                box_cropper = SpatialCropPadBox(roi_start=[0, 1, 4], roi_end=[21, 15, 8])
                boxes = torch.ones(2, 6)
                class_labels = torch.Tensor([0, 1])
                pred_scores = torch.Tensor([[0.4,0.3,0.3], [0.5,0.1,0.4]])
                labels = (class_labels, pred_scores)
                boxes_crop, labels_crop_tuple = box_cropper(boxes, labels)
        """
        spatial_dims = min(len(self.slices), get_spatial_dims(boxes=boxes))  # spatial dims
        boxes_crop, keep = spatial_crop_boxes(
            boxes,
            [self.slices[axis].start for axis in range(spatial_dims)],
            [self.slices[axis].stop for axis in range(spatial_dims)],
        )
        return boxes_crop, select_labels(labels, keep)


class RotateBox90(Rotate90):
    """
    Rotate a boxes by 90 degrees in the plane specified by `axes`.
    See box_ops.rot90_boxes for additional details

    Args:
        k: number of times to rotate by 90 degrees.
        spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
            Default: (0, 1), this is the first two axis in spatial dimensions.
            If axis is negative it counts from the last to the first axis.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, boxes: NdarrayTensor, spatial_size: Sequence[int] | int) -> NdarrayTensor:  # type: ignore[override]
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        return rot90_boxes(boxes, spatial_size, self.k, self.spatial_axes)
