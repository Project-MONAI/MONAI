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
A collection of "vanilla" transforms for crop and pad operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from itertools import chain
from math import ceil
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn.functional import pad as pad_pt

from monai.config import IndexSelection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.inverse import InvertibleTransform, TraceableTransform
from monai.transforms.transform import Randomizable, Transform
from monai.transforms.utils import (
    compute_divisible_spatial_size,
    convert_pad_mode,
    create_translate,
    generate_label_classes_crop_centers,
    generate_pos_neg_label_crop_centers,
    generate_spatial_bounding_box,
    is_positive,
    map_binary_to_indices,
    map_classes_to_indices,
    weighted_patch_samples,
)
from monai.utils import ImageMetaKey as Key
from monai.utils import (
    Method,
    PytorchPadMode,
    TraceKeys,
    TransformBackends,
    convert_data_type,
    convert_to_dst_type,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    pytorch_after,
)

__all__ = [
    "Pad",
    "SpatialPad",
    "BorderPad",
    "DivisiblePad",
    "Crop",
    "SpatialCrop",
    "CenterSpatialCrop",
    "CenterScaleCrop",
    "RandSpatialCrop",
    "RandScaleCrop",
    "RandSpatialCropSamples",
    "CropForeground",
    "RandWeightedCrop",
    "RandCropByPosNegLabel",
    "RandCropByLabelClasses",
    "ResizeWithPadOrCrop",
    "BoundingRect",
]


class Pad(InvertibleTransform):
    """
    Perform padding for a given an amount of padding in each dimension.

    `torch.nn.functional.pad` is used unless the mode or kwargs are not available in torch,
    in which case `np.pad` will be used.

    Args:
        to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
            if None, must provide in the `__call__` at runtime.
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self, to_pad: Optional[List[Tuple[int, int]]] = None, mode: str = PytorchPadMode.CONSTANT, **kwargs
    ) -> None:
        self.to_pad = to_pad
        self.mode = mode
        self.kwargs = kwargs

    def compute_pad_width(self, spatial_shape: Sequence[int]) -> List[Tuple[int, int]]:
        """
        dynamically compute the pad width according to the spatial shape.
        the output is the amount of padding for all dimensions including the channel.

        Args:
            spatial_shape: spatial shape of the original image.

        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @staticmethod
    def _np_pad(img: torch.Tensor, pad_width, mode, **kwargs) -> torch.Tensor:
        img_np = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img
        mode = convert_pad_mode(dst=img_np, mode=mode).value
        out = torch.as_tensor(np.pad(img, pad_width, mode=mode, **kwargs))
        if isinstance(img, MetaTensor):
            out = MetaTensor(out, meta=img.meta, applied_operations=img.applied_operations)
        return out

    @staticmethod
    def _pt_pad(img: torch.Tensor, pad_width, mode, **kwargs) -> torch.Tensor:
        pt_pad_width = [val for sublist in pad_width[1:] for val in sublist[::-1]][::-1]
        # torch.pad expects `[B, C, H, W, [D]]` shape
        return pad_pt(img.unsqueeze(0), pt_pad_width, mode=mode, **kwargs).squeeze(0)

    def __call__(  # type: ignore
        self, img: torch.Tensor, to_pad: Optional[List[Tuple[int, int]]] = None, mode: Optional[str] = None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and padding doesn't apply to the channel dim.
            to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
                default to `self.to_pad`.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        to_pad_ = self.to_pad if to_pad is None else to_pad
        if to_pad_ is None:
            to_pad_ = self.compute_pad_width(img.shape[1:])
        mode_ = self.mode if mode is None else mode
        kwargs_ = dict(self.kwargs)
        kwargs_.update(kwargs)

        img_t = convert_to_tensor(data=img, track_meta=get_track_meta())
        _orig_size = img_t.shape[1:]

        # all zeros, skip padding
        if np.asarray(to_pad_).any():
            if mode in ["linear_ramp", "maximum", "mean", "median", "minimum", "symmetric", "empty"]:
                out = self._np_pad(img_t, pad_width=to_pad_, mode=mode_, **kwargs_)
            else:
                try:
                    mode_ = convert_pad_mode(dst=img_t, mode=mode_).value
                    out = self._pt_pad(img_t, pad_width=to_pad_, mode=mode_, **kwargs_)
                # but if mode or args don't exist in pytorch, use numpy instead
                except (ValueError, TypeError) as err:
                    if "Unsupported option" in str(err) or "unexpected keyword" in str(err):
                        out = self._np_pad(img_t, pad_width=to_pad_, mode=mode_, **kwargs_)
        else:
            out = img_t
        if get_track_meta():
            self.update_meta(tensor=out, to_pad=to_pad_)  # type: ignore
            self.push_transform(out, orig_size=_orig_size, extra_info={"padded": to_pad_})
        return out

    def update_meta(self, tensor: MetaTensor, to_pad: List[Tuple[int, int]]):
        spatial_rank = max(len(tensor.affine) - 1, 1)
        to_shift = [-s[0] for s in to_pad[1:]]  # skipping the channel pad
        mat = create_translate(spatial_rank, to_shift)
        tensor.meta["affine"] = tensor.affine @ convert_to_dst_type(mat, tensor.affine)[0]

    def inverse(self, data: MetaTensor) -> MetaTensor:
        transform = self.pop_transform(data)
        padded = transform[TraceKeys.EXTRA_INFO]["padded"]
        if padded[0][0] > 0 or padded[0][1] > 0:  # slicing the channel dimension
            s = padded[0][0]
            e = min(max(padded[0][1], s + 1), len(data))
            data = data[s : len(data) - e]  # type: ignore
        roi_start = [i[0] for i in padded[1:]]
        roi_end = [i - j[1] for i, j in zip(data.shape[1:], padded[1:])]
        cropper = SpatialCrop(roi_start=roi_start, roi_end=roi_end)
        with cropper.trace_transform(False):
            return cropper(data)  # type: ignore


class SpatialPad(Pad):
    """
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.

    Args:
        spatial_size: the spatial size of output data after padding, if a dimension of the input
            data size is larger than the pad size, will not pad that dimension.
            If its components have non-positive values, the corresponding size of input image will be used
            (no padding). for example: if the spatial size of input data is [30, 30, 30] and
            `spatial_size=[32, 25, -1]`, the spatial size of output data will be [32, 30, 30].
        method: {``"symmetric"``, ``"end"``}
            Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        method: str = Method.SYMMETRIC,
        mode: str = PytorchPadMode.CONSTANT,
        **kwargs,
    ) -> None:
        self.spatial_size = spatial_size
        self.method: Method = look_up_option(method, Method)
        super().__init__(mode=mode, **kwargs)

    def compute_pad_width(self, spatial_shape: Sequence[int]) -> List[Tuple[int, int]]:
        """
        dynamically compute the pad width according to the spatial shape.

        Args:
            spatial_shape: spatial shape of the original image.

        """
        spatial_size = fall_back_tuple(self.spatial_size, spatial_shape)
        if self.method == Method.SYMMETRIC:
            pad_width = []
            for i, sp_i in enumerate(spatial_size):
                width = max(sp_i - spatial_shape[i], 0)
                pad_width.append((width // 2, width - (width // 2)))
        else:
            pad_width = [(0, max(sp_i - spatial_shape[i], 0)) for i, sp_i in enumerate(spatial_size)]
        return [(0, 0)] + pad_width


class BorderPad(Pad):
    """
    Pad the input data by adding specified borders to every dimension.

    Args:
        spatial_border: specified size for every spatial border. Any -ve values will be set to 0. It can be 3 shapes:

            - single int number, pad all the borders with the same size.
            - length equals the length of image shape, pad every spatial dimension separately.
              for example, image shape(CHW) is [1, 4, 4], spatial_border is [2, 1],
              pad every border of H dim with 2, pad every border of W dim with 1, result shape is [1, 8, 6].
            - length equals 2 x (length of image shape), pad every border of every dimension separately.
              for example, image shape(CHW) is [1, 4, 4], spatial_border is [1, 2, 3, 4], pad top of H dim with 1,
              pad bottom of H dim with 2, pad left of W dim with 3, pad right of W dim with 4.
              the result shape is [1, 7, 11].
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    def __init__(
        self, spatial_border: Union[Sequence[int], int], mode: str = PytorchPadMode.CONSTANT, **kwargs
    ) -> None:
        self.spatial_border = spatial_border
        super().__init__(mode=mode, **kwargs)

    def compute_pad_width(self, spatial_shape: Sequence[int]) -> List[Tuple[int, int]]:
        spatial_border = ensure_tuple(self.spatial_border)
        if not all(isinstance(b, int) for b in spatial_border):
            raise ValueError(f"self.spatial_border must contain only ints, got {spatial_border}.")
        spatial_border = tuple(max(0, b) for b in spatial_border)

        if len(spatial_border) == 1:
            data_pad_width = [(spatial_border[0], spatial_border[0]) for _ in spatial_shape]
        elif len(spatial_border) == len(spatial_shape):
            data_pad_width = [(sp, sp) for sp in spatial_border[: len(spatial_shape)]]
        elif len(spatial_border) == len(spatial_shape) * 2:
            data_pad_width = [(spatial_border[2 * i], spatial_border[2 * i + 1]) for i in range(len(spatial_shape))]
        else:
            raise ValueError(
                f"Unsupported spatial_border length: {len(spatial_border)}, available options are "
                f"[1, len(spatial_shape)={len(spatial_shape)}, 2*len(spatial_shape)={2*len(spatial_shape)}]."
            )
        return [(0, 0)] + data_pad_width


class DivisiblePad(Pad):
    """
    Pad the input data, so that the spatial sizes are divisible by `k`.
    """

    backend = SpatialPad.backend

    def __init__(
        self,
        k: Union[Sequence[int], int],
        mode: str = PytorchPadMode.CONSTANT,
        method: str = Method.SYMMETRIC,
        **kwargs,
    ) -> None:
        """
        Args:
            k: the target k for each spatial dimension.
                if `k` is negative or 0, the original size is preserved.
                if `k` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            method: {``"symmetric"``, ``"end"``}
                Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
            kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        See also :py:class:`monai.transforms.SpatialPad`
        """
        self.k = k
        self.method: Method = Method(method)
        super().__init__(mode=mode, **kwargs)

    def compute_pad_width(self, spatial_shape: Sequence[int]) -> List[Tuple[int, int]]:
        new_size = compute_divisible_spatial_size(spatial_shape=spatial_shape, k=self.k)
        spatial_pad = SpatialPad(spatial_size=new_size, method=self.method)
        return spatial_pad.compute_pad_width(spatial_shape)


class Crop(InvertibleTransform):
    """
    Perform crop operation on the input image.

    """

    backend = [TransformBackends.TORCH]

    @staticmethod
    def compute_slices(
        roi_center: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_size: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_start: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_end: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_slices: Optional[Sequence[slice]] = None,
    ):
        """
        Compute the crop slices based on specified `center & size` or `start & end`.

        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is larger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.

        """
        roi_start_t: torch.Tensor

        if roi_slices:
            if not all(s.step is None or s.step == 1 for s in roi_slices):
                raise ValueError("only slice steps of 1/None are currently supported")
            return list(roi_slices)
        else:
            if roi_center is not None and roi_size is not None:
                roi_center_t = convert_to_tensor(data=roi_center, dtype=torch.int16, wrap_sequence=True)
                roi_size_t = convert_to_tensor(data=roi_size, dtype=torch.int16, wrap_sequence=True)
                _zeros = torch.zeros_like(roi_center_t)
                half = (
                    torch.divide(roi_size_t, 2, rounding_mode="floor")
                    if pytorch_after(1, 8)
                    else torch.floor_divide(roi_size_t, 2)
                )
                roi_start_t = torch.maximum(roi_center_t - half, _zeros)
                roi_end_t = torch.maximum(roi_start_t + roi_size_t, roi_start_t)
            else:
                if roi_start is None or roi_end is None:
                    raise ValueError("please specify either roi_center, roi_size or roi_start, roi_end.")
                roi_start_t = convert_to_tensor(data=roi_start, dtype=torch.int16, wrap_sequence=True)
                roi_start_t = torch.maximum(roi_start_t, torch.zeros_like(roi_start_t))
                roi_end_t = convert_to_tensor(data=roi_end, dtype=torch.int16, wrap_sequence=True)
                roi_end_t = torch.maximum(roi_end_t, roi_start_t)
            # convert to slices (accounting for 1d)
            if roi_start_t.numel() == 1:
                return [slice(int(roi_start_t.item()), int(roi_end_t.item()))]
            else:
                return [slice(int(s), int(e)) for s, e in zip(roi_start_t.tolist(), roi_end_t.tolist())]

    def __call__(self, img: torch.Tensor, slices: Tuple[slice, ...]) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        orig_size = img.shape[1:]
        slices_ = list(slices)
        sd = len(img.shape[1:])  # spatial dims
        if len(slices_) < sd:
            slices_ += [slice(None)] * (sd - len(slices_))
        # Add in the channel (no cropping)
        slices = tuple([slice(None)] + slices_[:sd])

        img_t: MetaTensor = convert_to_tensor(data=img, track_meta=get_track_meta())
        _orig_size = img_t.shape[1:]
        img_t = img_t[slices]  # type: ignore
        if get_track_meta():
            self.update_meta(tensor=img_t, slices=slices)
            cropped_from_start = np.asarray([s.indices(o)[0] for s, o in zip(slices[1:], orig_size)])
            cropped_from_end = np.asarray(orig_size) - img_t.shape[1:] - cropped_from_start
            cropped = list(chain(*zip(cropped_from_start.tolist(), cropped_from_end.tolist())))
            self.push_transform(img_t, orig_size=_orig_size, extra_info={"cropped": cropped})
        return img_t

    def update_meta(self, tensor: MetaTensor, slices: Tuple[slice, ...]):
        spatial_rank = max(len(tensor.affine) - 1, 1)
        to_shift = [s.start if s.start is not None else 0 for s in ensure_tuple(slices)[1:]]
        mat = create_translate(spatial_rank, to_shift)
        tensor.meta["affine"] = tensor.affine @ convert_to_dst_type(mat, tensor.affine)[0]

    def inverse(self, img: MetaTensor) -> MetaTensor:
        transform = self.pop_transform(img)
        cropped = transform[TraceKeys.EXTRA_INFO]["cropped"]
        # the amount we pad is equal to the amount we cropped in each direction
        inverse_transform = BorderPad(cropped)
        # Apply inverse transform
        with inverse_transform.trace_transform(False):
            return inverse_transform(img)  # type: ignore


class SpatialCrop(Crop):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    If a dimension of the expected ROI size is larger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial (channel-first) data.

    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (allows for use of -ve indexing and `None`)
        - a spatial center and size
        - the start and end coordinates of the ROI
    """

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
            roi_size: size of the crop ROI, if a dimension of ROI size is larger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
        """
        self.slices = self.compute_slices(
            roi_center=roi_center, roi_size=roi_size, roi_start=roi_start, roi_end=roi_end, roi_slices=roi_slices
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        return super().__call__(img=img, slices=self.slices)


class CenterSpatialCrop(Crop):
    """
    Crop at the center of image with specified ROI size.
    If a dimension of the expected ROI size is larger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
    """

    def __init__(self, roi_size: Union[Sequence[int], int]) -> None:
        self.roi_size = roi_size

    def compute_slices(self, spatial_size: Sequence[int]):  # type: ignore
        roi_size = fall_back_tuple(self.roi_size, spatial_size)
        roi_center = [i // 2 for i in spatial_size]
        return super().compute_slices(roi_center=roi_center, roi_size=roi_size)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        return super().__call__(img=img, slices=self.compute_slices(img.shape[1:]))


class CenterScaleCrop(Crop):
    """
    Crop at the center of image with specified scale of ROI size.

    Args:
        roi_scale: specifies the expected scale of image size to crop. e.g. [0.3, 0.4, 0.5] or a number for all dims.
            If its components have non-positive values, will use `1.0` instead, which means the input image size.

    """

    def __init__(self, roi_scale: Union[Sequence[float], float]):
        self.roi_scale = roi_scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:  # type: ignore
        img_size = img.shape[1:]
        ndim = len(img_size)
        roi_size = [ceil(r * s) for r, s in zip(ensure_tuple_rep(self.roi_scale, ndim), img_size)]
        cropper = CenterSpatialCrop(roi_size=roi_size)
        return super().__call__(img=img, slices=cropper.compute_slices(img.shape[1:]))


class RandSpatialCrop(Randomizable, Crop):
    """
    Crop image with random size or specific size ROI. It can crop at a random position as center
    or at the image center. And allows to set the minimum and maximum size to limit the randomly generated ROI.

    Note: even `random_size=False`, if a dimension of the expected ROI size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped results
    of several images may not have exactly the same shape.

    Args:
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        max_roi_size: if `random_size` is True and `roi_size` specifies the min crop region size, `max_roi_size`
            can specify the max crop region size. if None, defaults to the input image size.
            if its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            if True, the actual size is sampled from `randint(roi_size, max_roi_size + 1)`.
    """

    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        max_roi_size: Optional[Union[Sequence[int], int]] = None,
        random_center: bool = True,
        random_size: bool = True,
    ) -> None:
        self.roi_size = roi_size
        self.max_roi_size = max_roi_size
        self.random_center = random_center
        self.random_size = random_size
        self._size: Optional[Sequence[int]] = None
        self._slices: Tuple[slice, ...]

    def randomize(self, img_size: Sequence[int]) -> None:
        self._size = fall_back_tuple(self.roi_size, img_size)
        if self.random_size:
            max_size = img_size if self.max_roi_size is None else fall_back_tuple(self.max_roi_size, img_size)
            if any(i > j for i, j in zip(self._size, max_size)):
                raise ValueError(f"min ROI size: {self._size} is larger than max ROI size: {max_size}.")
            self._size = tuple(self.R.randint(low=self._size[i], high=max_size[i] + 1) for i in range(len(img_size)))
        if self.random_center:
            valid_size = get_valid_patch_size(img_size, self._size)
            self._slices = get_random_patch(img_size, valid_size, self.R)

    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        if randomize:
            self.randomize(img.shape[1:])
        if self._size is None:
            raise RuntimeError("self._size not specified.")
        if self.random_center:
            return super().__call__(img=img, slices=self._slices)
        cropper = CenterSpatialCrop(self._size)
        return super().__call__(img=img, slices=cropper.compute_slices(img.shape[1:]))


class RandScaleCrop(RandSpatialCrop):
    """
    Subclass of :py:class:`monai.transforms.RandSpatialCrop`. Crop image with
    random size or specific size ROI.  It can crop at a random position as
    center or at the image center.  And allows to set the minimum and maximum
    scale of image size to limit the randomly generated ROI.

    Args:
        roi_scale: if `random_size` is True, it specifies the minimum crop size: `roi_scale * image spatial size`.
            if `random_size` is False, it specifies the expected scale of image size to crop. e.g. [0.3, 0.4, 0.5].
            If its components have non-positive values, will use `1.0` instead, which means the input image size.
        max_roi_scale: if `random_size` is True and `roi_scale` specifies the min crop region size, `max_roi_scale`
            can specify the max crop region size: `max_roi_scale * image spatial size`.
            if None, defaults to the input image size. if its components have non-positive values,
            will use `1.0` instead, which means the input image size.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specified size ROI by `roi_scale * image spatial size`.
            if True, the actual size is sampled from
            `randint(roi_scale * image spatial size, max_roi_scale * image spatial size + 1)`.
    """

    def __init__(
        self,
        roi_scale: Union[Sequence[float], float],
        max_roi_scale: Optional[Union[Sequence[float], float]] = None,
        random_center: bool = True,
        random_size: bool = True,
    ) -> None:
        super().__init__(roi_size=-1, max_roi_size=None, random_center=random_center, random_size=random_size)
        self.roi_scale = roi_scale
        self.max_roi_scale = max_roi_scale

    def get_max_roi_size(self, img_size):
        ndim = len(img_size)
        self.roi_size = [ceil(r * s) for r, s in zip(ensure_tuple_rep(self.roi_scale, ndim), img_size)]
        if self.max_roi_scale is not None:
            self.max_roi_size = [ceil(r * s) for r, s in zip(ensure_tuple_rep(self.max_roi_scale, ndim), img_size)]
        else:
            self.max_roi_size = None

    def randomize(self, img_size: Sequence[int]) -> None:
        self.get_max_roi_size(img_size)
        super().randomize(img_size)

    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        self.get_max_roi_size(img.shape[1:])
        return super().__call__(img=img, randomize=randomize)


class RandSpatialCropSamples(Randomizable, TraceableTransform):
    """
    Crop image with random size or specific size ROI to generate a list of N samples.
    It can crop at a random position as center or at the image center. And allows to set
    the minimum size to limit the randomly generated ROI.
    It will return a list of cropped images.

    Note: even `random_size=False`, if a dimension of the expected ROI size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped
    results of several images may not have exactly the same shape.

    Args:
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        num_samples: number of samples (crop regions) to take in the returned list.
        max_roi_size: if `random_size` is True and `roi_size` specifies the min crop region size, `max_roi_size`
            can specify the max crop region size. if None, defaults to the input image size.
            if its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.

    Raises:
        ValueError: When ``num_samples`` is nonpositive.

    """

    backend = RandSpatialCrop.backend

    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        num_samples: int,
        max_roi_size: Optional[Union[Sequence[int], int]] = None,
        random_center: bool = True,
        random_size: bool = True,
    ) -> None:
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self.num_samples = num_samples
        self.cropper = RandSpatialCrop(roi_size, max_roi_size, random_center, random_size)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSpatialCropSamples":
        super().set_random_state(seed, state)
        self.cropper.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        pass

    def __call__(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        cropping doesn't change the channel dim.
        """
        ret = []
        orig_size = img.shape[1:]
        for i in range(self.num_samples):
            cropped = self.cropper(img)
            if get_track_meta():
                cropped.meta[Key.PATCH_INDEX] = i  # type: ignore
                self.push_transform(cropped, orig_size=orig_size, extra_info=self.pop_transform(cropped, check=False))
            ret.append(cropped)
        return ret


class CropForeground(Crop):
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

    def __init__(
        self,
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int] = 0,
        allow_smaller: bool = True,
        return_coords: bool = False,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ) -> None:
        """
        Args:
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            allow_smaller: when computing box size with `margin`, whether allow the image size to be smaller
                than box size, default to `True`. if the margined size is larger than image size, will pad with
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
            pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin
        self.allow_smaller = allow_smaller
        self.return_coords = return_coords
        self.k_divisible = k_divisible
        self.padder = Pad(mode=mode, **pad_kwargs)

    def compute_bounding_box(self, img: torch.Tensor):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = generate_spatial_bounding_box(
            img, self.select_fn, self.channel_indices, self.margin, self.allow_smaller
        )
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
        self, img: torch.Tensor, box_start: np.ndarray, box_end: np.ndarray, mode: Optional[str] = None, **pad_kwargs
    ):
        """
        Crop and pad based on the bounding box.

        """
        slices = self.compute_slices(roi_start=box_start, roi_end=box_end)
        cropped = super().__call__(img=img, slices=slices)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(box_end - np.asarray(img.shape[1:]), 0)
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        pad_width = BorderPad(spatial_border=pad).compute_pad_width(cropped.shape[1:])
        ret = self.padder.__call__(img=cropped, to_pad=pad_width, mode=mode, **pad_kwargs)
        # combine the traced cropping and padding into one transformation
        # by taking the padded info and placing it in a key inside the crop info.
        if get_track_meta():
            ret_: MetaTensor = ret  # type: ignore
            app_op = ret_.applied_operations.pop(-1)
            ret_.applied_operations[-1][TraceKeys.EXTRA_INFO]["pad_info"] = app_op
        return ret

    def __call__(self, img: torch.Tensor, mode: Optional[str] = None, **pad_kwargs):  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        box_start, box_end = self.compute_bounding_box(img)
        cropped = self.crop_pad(img, box_start, box_end, mode, **pad_kwargs)

        if self.return_coords:
            return cropped, box_start, box_end
        return cropped

    def inverse(self, img: MetaTensor) -> MetaTensor:
        transform = self.get_most_recent_transform(img)
        # we moved the padding info in the forward, so put it back for the inverse
        pad_info = transform[TraceKeys.EXTRA_INFO].pop("pad_info")
        img.applied_operations.append(pad_info)
        # first inverse the padder
        inv = self.padder.inverse(img)
        # and then inverse the cropper (self)
        return super().inverse(inv)


class RandWeightedCrop(Randomizable, TraceableTransform):
    """
    Samples a list of `num_samples` image patches according to the provided `weight_map`.

    Args:
        spatial_size: the spatial size of the image patch e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `img` will be used.
        num_samples: number of samples (image patches) to take in the returned list.
        weight_map: weight map used to generate patch samples. The weights must be non-negative.
            Each element denotes a sampling weight of the spatial location. 0 indicates no sampling.
            It should be a single-channel array in shape, for example, `(1, spatial_dim_0, spatial_dim_1, ...)`.
    """

    backend = SpatialCrop.backend

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        num_samples: int = 1,
        weight_map: Optional[NdarrayOrTensor] = None,
    ):
        self.spatial_size = ensure_tuple(spatial_size)
        self.num_samples = int(num_samples)
        self.weight_map = weight_map
        self.centers: List[np.ndarray] = []

    def randomize(self, weight_map: NdarrayOrTensor) -> None:
        self.centers = weighted_patch_samples(
            spatial_size=self.spatial_size, w=weight_map[0], n_samples=self.num_samples, r_state=self.R
        )  # using only the first channel as weight map

    def __call__(
        self, img: torch.Tensor, weight_map: Optional[NdarrayOrTensor] = None, randomize: bool = True
    ) -> List[torch.Tensor]:
        """
        Args:
            img: input image to sample patches from. assuming `img` is a channel-first array.
            weight_map: weight map used to generate patch samples. The weights must be non-negative.
                Each element denotes a sampling weight of the spatial location. 0 indicates no sampling.
                It should be a single-channel array in shape, for example, `(1, spatial_dim_0, spatial_dim_1, ...)`
            randomize: whether to execute random operations, default to `True`.

        Returns:
            A list of image patches
        """
        if weight_map is None:
            weight_map = self.weight_map
        if weight_map is None:
            raise ValueError("weight map must be provided for weighted patch sampling.")
        if img.shape[1:] != weight_map.shape[1:]:
            raise ValueError(f"image and weight map spatial shape mismatch: {img.shape[1:]} vs {weight_map.shape[1:]}.")

        if randomize:
            self.randomize(weight_map)
        _spatial_size = fall_back_tuple(self.spatial_size, weight_map.shape[1:])
        results: List[torch.Tensor] = []
        orig_size = img.shape[1:]
        for i, center in enumerate(self.centers):
            cropped = SpatialCrop(roi_center=center, roi_size=_spatial_size)(img)
            if get_track_meta():
                ret_: MetaTensor = cropped  # type: ignore
                ret_.meta[Key.PATCH_INDEX] = i
                ret_.meta["crop_center"] = center
                self.push_transform(ret_, orig_size=orig_size, extra_info=self.pop_transform(ret_, check=False))
            results.append(cropped)
        return results


class RandCropByPosNegLabel(Randomizable, TraceableTransform):
    """
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::

        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]

    If a dimension of the expected spatial size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center to ensure the
    valid crop ROI.

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    backend = SpatialCrop.backend

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        label: Optional[torch.Tensor] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: Optional[torch.Tensor] = None,
        image_threshold: float = 0.0,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        allow_smaller: bool = False,
    ) -> None:
        self.spatial_size = spatial_size
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[int]]] = None
        self.fg_indices = fg_indices
        self.bg_indices = bg_indices
        self.allow_smaller = allow_smaller

    def randomize(
        self,
        label: torch.Tensor,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> None:
        if fg_indices is None or bg_indices is None:
            if self.fg_indices is not None and self.bg_indices is not None:
                fg_indices_ = self.fg_indices
                bg_indices_ = self.bg_indices
            else:
                fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size,
            self.num_samples,
            self.pos_ratio,
            label.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
            self.allow_smaller,
        )

    def __call__(
        self,
        img: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        randomize: bool = True,
    ) -> List[torch.Tensor]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            randomize: whether to execute the random operations, default to `True`.

        """
        if label is None:
            label = self.label
        if label is None:
            raise ValueError("label should be provided.")
        if image is None:
            image = self.image

        if randomize:
            self.randomize(label, fg_indices, bg_indices, image)
        results: List[torch.Tensor] = []
        orig_size = img.shape[1:]
        if self.centers is not None:
            for i, center in enumerate(self.centers):
                roi_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
                cropped = SpatialCrop(roi_center=center, roi_size=roi_size)(img)
                if get_track_meta():
                    ret_: MetaTensor = cropped  # type: ignore
                    ret_.meta[Key.PATCH_INDEX] = i
                    ret_.meta["crop_center"] = center
                    self.push_transform(ret_, orig_size=orig_size, extra_info=self.pop_transform(ret_, check=False))
                results.append(cropped)
        return results


class RandCropByLabelClasses(Randomizable, TraceableTransform):
    """
    Crop random fixed sized regions with the center being a class based on the specified ratios of every class.
    The label data can be One-Hot format array or Argmax data. And will return a list of arrays for all the
    cropped images. For example, crop two (3 x 3) arrays from (5 x 5) array with `ratios=[1, 2, 3, 1]`::

        image = np.array([
            [[0.0, 0.3, 0.4, 0.2, 0.0],
            [0.0, 0.1, 0.2, 0.1, 0.4],
            [0.0, 0.3, 0.5, 0.2, 0.0],
            [0.1, 0.2, 0.1, 0.1, 0.0],
            [0.0, 0.1, 0.2, 0.1, 0.0]]
        ])
        label = np.array([
            [[0, 0, 0, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]
        ])
        cropper = RandCropByLabelClasses(
            spatial_size=[3, 3],
            ratios=[1, 2, 3, 1],
            num_classes=4,
            num_samples=2,
        )
        label_samples = cropper(img=label, label=label, image=image)

        The 2 randomly cropped samples of `label` can be:
        [[0, 1, 2],     [[0, 0, 0],
         [0, 1, 3],      [1, 2, 1],
         [0, 0, 0]]      [1, 3, 0]]

    If a dimension of the expected spatial size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center to ensure the
    valid crop ROI.

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        ratios: specified ratios of every class in the label to generate crop centers, including background class.
            if None, every class will have the same ratio to generate crop centers.
        label: the label image that is used for finding every classes, if None, must set at `self.__call__`.
        num_classes: number of classes for argmax label, not necessary for One-Hot label.
        num_samples: number of samples (crop regions) to take in each list.
        image: if image is not None, only return the indices of every class that are within the valid
            region of the image (``image > image_threshold``).
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select class indices only in this area.
        indices: if provided pre-computed indices of every class, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, expect to be 1 dim array
            of spatial indices after flattening. a typical usage is to call `ClassesToIndices` transform first
            and cache the results for better performance.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will remain
            unchanged.

    """

    backend = SpatialCrop.backend

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        ratios: Optional[List[Union[float, int]]] = None,
        label: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
        num_samples: int = 1,
        image: Optional[torch.Tensor] = None,
        image_threshold: float = 0.0,
        indices: Optional[List[NdarrayOrTensor]] = None,
        allow_smaller: bool = False,
    ) -> None:
        self.spatial_size = spatial_size
        self.ratios = ratios
        self.label = label
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[int]]] = None
        self.indices = indices
        self.allow_smaller = allow_smaller

    def randomize(
        self, label: torch.Tensor, indices: Optional[List[NdarrayOrTensor]] = None, image: Optional[torch.Tensor] = None
    ) -> None:
        indices_: Sequence[NdarrayOrTensor]
        if indices is None:
            if self.indices is not None:
                indices_ = self.indices
            else:
                indices_ = map_classes_to_indices(label, self.num_classes, image, self.image_threshold)
        else:
            indices_ = indices
        self.centers = generate_label_classes_crop_centers(
            self.spatial_size, self.num_samples, label.shape[1:], indices_, self.ratios, self.R, self.allow_smaller
        )

    def __call__(
        self,
        img: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        indices: Optional[List[NdarrayOrTensor]] = None,
        randomize: bool = True,
    ) -> List[torch.Tensor]:
        """
        Args:
            img: input data to crop samples from based on the ratios of every class, assumes `img` is a
                channel-first array.
            label: the label image that is used for finding indices of every class, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``image > image_threshold`` to select the centers only in valid region. if None, use `self.image`.
            indices: list of indices for every class in the image, used to randomly select crop centers.
            randomize: whether to execute the random operations, default to `True`.

        """
        if label is None:
            label = self.label
        if label is None:
            raise ValueError("label should be provided.")
        if image is None:
            image = self.image

        if randomize:
            self.randomize(label, indices, image)
        results: List[torch.Tensor] = []
        orig_size = img.shape[1:]
        if self.centers is not None:
            for i, center in enumerate(self.centers):
                roi_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
                cropped = SpatialCrop(roi_center=tuple(center), roi_size=roi_size)(img)
                if get_track_meta():
                    ret_: MetaTensor = cropped  # type: ignore
                    ret_.meta[Key.PATCH_INDEX] = i
                    ret_.meta["crop_center"] = center
                    self.push_transform(ret_, orig_size=orig_size, extra_info=self.pop_transform(ret_, check=False))
                results.append(cropped)

        return results


class ResizeWithPadOrCrop(InvertibleTransform):
    """
    Resize an image to a target spatial size by either centrally cropping the image or
    padding it evenly with a user-specified mode.
    When the dimension is smaller than the target size, do symmetric padding along that dim.
    When the dimension is larger than the target size, do central cropping along that dim.

    Args:
        spatial_size: the spatial size of output data after padding or crop.
            If has non-positive values, the corresponding size of input image will be used (no padding).
        method: {``"symmetric"``, ``"end"``}
            Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = list(set(SpatialPad.backend) & set(CenterSpatialCrop.backend))

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        method: str = Method.SYMMETRIC,
        mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ):
        self.padder = SpatialPad(spatial_size=spatial_size, method=method, mode=mode, **pad_kwargs)
        self.cropper = CenterSpatialCrop(roi_size=spatial_size)

    def __call__(self, img: torch.Tensor, mode: Optional[str] = None, **pad_kwargs) -> torch.Tensor:  # type: ignore
        """
        Args:
            img: data to pad or crop, assuming `img` is channel-first and
                padding or cropping doesn't apply to the channel dim.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        orig_size = img.shape[1:]
        ret = self.padder(self.cropper(img), mode=mode, **pad_kwargs)
        # remove the individual info and combine
        if get_track_meta():
            ret_: MetaTensor = ret  # type: ignore
            pad_info = ret_.applied_operations.pop(-1)
            crop_info = ret_.applied_operations.pop(-1)
            self.push_transform(ret_, orig_size=orig_size, extra_info={"pad_info": pad_info, "crop_info": crop_info})
        return ret

    def inverse(self, img: MetaTensor) -> MetaTensor:
        transform = self.pop_transform(img)
        return self.inverse_transform(img, transform)

    def inverse_transform(self, img: MetaTensor, transform) -> MetaTensor:
        # we joined the cropping and padding, so put them back before calling the inverse
        crop_info = transform[TraceKeys.EXTRA_INFO].pop("crop_info")
        pad_info = transform[TraceKeys.EXTRA_INFO].pop("pad_info")
        img.applied_operations.append(crop_info)
        img.applied_operations.append(pad_info)
        # first inverse the padder
        inv = self.padder.inverse(img)
        # and then inverse the cropper (self)
        return self.cropper.inverse(inv)


class BoundingRect(Transform):
    """
    Compute coordinates of axis-aligned bounding rectangles from input image `img`.
    The output format of the coordinates is (shape is [channel, 2 * spatial dims]):

        [[1st_spatial_dim_start, 1st_spatial_dim_end,
         2nd_spatial_dim_start, 2nd_spatial_dim_end,
         ...,
         Nth_spatial_dim_start, Nth_spatial_dim_end],

         ...

         [1st_spatial_dim_start, 1st_spatial_dim_end,
         2nd_spatial_dim_start, 2nd_spatial_dim_end,
         ...,
         Nth_spatial_dim_start, Nth_spatial_dim_end]]

    The bounding boxes edges are aligned with the input image edges.
    This function returns [0, 0, ...] if there's no positive intensity.

    Args:
        select_fn: function to select expected foreground, default is to select values > 0.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, select_fn: Callable = is_positive) -> None:
        self.select_fn = select_fn

    def __call__(self, img: NdarrayOrTensor) -> np.ndarray:
        """
        See also: :py:class:`monai.transforms.utils.generate_spatial_bounding_box`.
        """
        bbox = []

        for channel in range(img.shape[0]):
            start_, end_ = generate_spatial_bounding_box(img, select_fn=self.select_fn, channel_indices=channel)
            bbox.append([i for k in zip(start_, end_) for i in k])

        return np.stack(bbox, axis=0)
