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
A collection of dictionary-based wrappers around the "vanilla" transforms for crop and pad operations
defined in :py:class:`monai.transforms.croppad.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

import contextlib
from copy import deepcopy
from enum import Enum
from itertools import chain
from math import floor
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import IndexSelection, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.croppad.array import (
    BorderPad,
    BoundingRect,
    CenterSpatialCrop,
    CropBase,
    CropForeground,
    DivisiblePad,
    PadBase,
    RandCropByLabelClasses,
    RandCropByPosNegLabel,
    ResizeWithPadOrCrop,
    SpatialCrop,
    SpatialPad,
)
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.utils import (
    allow_missing_keys_mode,
    generate_label_classes_crop_centers,
    generate_pos_neg_label_crop_centers,
    is_positive,
    map_binary_to_indices,
    map_classes_to_indices,
    weighted_patch_samples,
)
from monai.utils import ImageMetaKey as Key
from monai.utils import Method, NumpyPadMode, PytorchPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple
from monai.utils.enums import PostFix, TraceKeys

__all__ = [
    "PadModeSequence",
    "PadBased",
    "SpatialPadd",
    "BorderPadd",
    "DivisiblePadd",
    "SpatialCropd",
    "CenterSpatialCropd",
    "CenterScaleCropd",
    "RandScaleCropd",
    "RandSpatialCropd",
    "RandSpatialCropSamplesd",
    "CropForegroundd",
    "RandWeightedCropd",
    "RandCropByPosNegLabeld",
    "ResizeWithPadOrCropd",
    "BoundingRectd",
    "RandCropByLabelClassesd",
    "SpatialPadD",
    "SpatialPadDict",
    "BorderPadD",
    "BorderPadDict",
    "DivisiblePadD",
    "DivisiblePadDict",
    "SpatialCropD",
    "SpatialCropDict",
    "CenterSpatialCropD",
    "CenterSpatialCropDict",
    "CenterScaleCropD",
    "CenterScaleCropDict",
    "RandScaleCropD",
    "RandScaleCropDict",
    "RandSpatialCropD",
    "RandSpatialCropDict",
    "RandSpatialCropSamplesD",
    "RandSpatialCropSamplesDict",
    "CropForegroundD",
    "CropForegroundDict",
    "RandWeightedCropD",
    "RandWeightedCropDict",
    "RandCropByPosNegLabelD",
    "RandCropByPosNegLabelDict",
    "ResizeWithPadOrCropD",
    "ResizeWithPadOrCropDict",
    "BoundingRectD",
    "BoundingRectDict",
    "RandCropByLabelClassesD",
    "RandCropByLabelClassesDict",
]

NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]
PadModeSequence = Union[Sequence[Union[NumpyPadMode, PytorchPadMode, str]], NumpyPadMode, PytorchPadMode, str]
DEFAULT_POST_FIX = PostFix.meta()


class PadBased(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.PadBase`.
    """

    backend = PadBase.backend
    padder: PadBase

    def __init__(
        self, keys: KeysCollection, mode: PadModeSequence = NumpyPadMode.CONSTANT, allow_missing_keys: bool = False
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        if self.padder is None:
            raise RuntimeError("PadBased should not be called directly, please use an inherited class.")
        d = dict(data)
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.padder(d[key], mode=m)
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.padder.inverse(d[key])
        return d


class SpatialPadd(PadBased):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialPad`.
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.
    """

    backend = SpatialPad.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        method: Union[Method, str] = Method.SYMMETRIC,
        mode: PadModeSequence = NumpyPadMode.CONSTANT,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            spatial_size: the spatial size of output data after padding, if a dimension of the input
                data size is bigger than the pad size, will not pad that dimension.
                If its components have non-positive values, the corresponding size of input image will be used.
                for example: if the spatial size of input data is [30, 30, 30] and `spatial_size=[32, 25, -1]`,
                the spatial size of output data will be [32, 30, 30].
            method: {``"symmetric"``, ``"end"``}
                Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.
            kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        super().__init__(keys, mode, allow_missing_keys)
        self.padder = SpatialPad(spatial_size, method, **kwargs)


class BorderPadd(PadBased):
    """
    Pad the input data by adding specified borders to every dimension.
    Dictionary-based wrapper of :py:class:`monai.transforms.BorderPad`.
    """

    backend = BorderPad.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_border: Union[Sequence[int], int],
        mode: PadModeSequence = NumpyPadMode.CONSTANT,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            spatial_border: specified size for every spatial border. it can be 3 shapes:

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
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.
            kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        super().__init__(keys, mode, allow_missing_keys)
        self.padder = BorderPad(spatial_border=spatial_border, **kwargs)


class DivisiblePadd(PadBased):
    """
    Pad the input data, so that the spatial sizes are divisible by `k`.
    Dictionary-based wrapper of :py:class:`monai.transforms.DivisiblePad`.
    """

    backend = DivisiblePad.backend

    def __init__(
        self,
        keys: KeysCollection,
        k: Union[Sequence[int], int],
        mode: PadModeSequence = NumpyPadMode.CONSTANT,
        method: Union[Method, str] = Method.SYMMETRIC,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            k: the target k for each spatial dimension.
                if `k` is negative or 0, the original size is preserved.
                if `k` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/ghttps://pytorch.orgenerated/numpy.pad.html
                /docs/stable/generated/torch.nn.functional.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            method: {``"symmetric"``, ``"end"``}
                Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
            allow_missing_keys: don't raise exception if key is missing.
            kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        See also :py:class:`monai.transforms.SpatialPad`

        """
        super().__init__(keys, mode, allow_missing_keys)
        self.padder = DivisiblePad(k=k, method=method, **kwargs)


class CropBased(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of abstract class :py:class:`monai.transforms.CropBase`.
    """
    backend = CropBase.backend
    cropper: CropBase

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.cropper(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.cropper.inverse(d[key])
        return d


class SpatialCropd(CropBased):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialCrop`.
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

    backend = SpatialCrop.backend

    def __init__(
        self,
        keys: KeysCollection,
        roi_center: Optional[Sequence[int]] = None,
        roi_size: Optional[Sequence[int]] = None,
        roi_start: Optional[Sequence[int]] = None,
        roi_end: Optional[Sequence[int]] = None,
        roi_slices: Optional[Sequence[slice]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.cropper = SpatialCrop(roi_center, roi_size, roi_start, roi_end, roi_slices)

class CenterSpatialCropd(CropBased):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CenterSpatialCrop`.
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: the size of the crop region e.g. [224,224,128]
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = CenterSpatialCrop.backend

    def __init__(
        self, keys: KeysCollection, roi_size: Union[Sequence[int], int], allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.cropper = CenterSpatialCrop(roi_size)

class CenterScaleCropd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CenterScaleCrop`.
    Note: as using the same scaled ROI to crop, all the input data specified by `keys` should have
    the same spatial shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_scale: specifies the expected scale of image size to crop. e.g. [0.3, 0.4, 0.5] or a number for all dims.
            If its components have non-positive values, will use `1.0` instead, which means the input image size.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = CenterSpatialCrop.backend

    def __init__(
        self, keys: KeysCollection, roi_scale: Union[Sequence[float], float], allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.roi_scale = roi_scale


class RandSpatialCropd(Randomizable, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandSpatialCrop`.
    Crop image with random size or specific size ROI. It can crop at a random position as
    center or at the image center. And allows to set the minimum and maximum size to limit the randomly
    generated ROI. Suppose all the expected fields specified by `keys` have same shape.

    Note: even `random_size=False`, if a dimension of the expected ROI size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped
    results of several images may not have exactly the same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        max_roi_size: if `random_size` is True and `roi_size` specifies the min crop region size, `max_roi_size`
            can specify the max crop region size. if None, defaults to the input image size.
            if its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            if True, the actual size is sampled from:
            `randint(roi_scale * image spatial size, max_roi_scale * image spatial size + 1)`.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = CenterSpatialCrop.backend

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Union[Sequence[int], int],
        max_roi_size: Optional[Union[Sequence[int], int]] = None,
        random_center: bool = True,
        random_size: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.roi_size = roi_size
        self.max_roi_size = max_roi_size
        self.random_center = random_center
        self.random_size = random_size
        self._slices: Optional[Tuple[slice, ...]] = None
        self._size: Optional[Sequence[int]] = None


class RandScaleCropd(RandSpatialCropd):
    """
    Dictionary-based version :py:class:`monai.transforms.RandScaleCrop`.
    Crop image with random size or specific size ROI.
    It can crop at a random position as center or at the image center.
    And allows to set the minimum and maximum scale of image size to limit the randomly generated ROI.
    Suppose all the expected fields specified by `keys` have same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_scale: if `random_size` is True, it specifies the minimum crop size: `roi_scale * image spatial size`.
            if `random_size` is False, it specifies the expected scale of image size to crop. e.g. [0.3, 0.4, 0.5].
            If its components have non-positive values, will use `1.0` instead, which means the input image size.
        max_roi_scale: if `random_size` is True and `roi_scale` specifies the min crop region size, `max_roi_scale`
            can specify the max crop region size: `max_roi_scale * image spatial size`.
            if None, defaults to the input image size. if its components have non-positive values,
            will use `1.0` instead, which means the input image size.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specified size ROI by `roi_scale * image spatial size`.
            if True, the actual size is sampled from:
            `randint(roi_scale * image spatial size, max_roi_scale * image spatial size + 1)`.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandSpatialCropd.backend

    def __init__(
        self,
        keys: KeysCollection,
        roi_scale: Union[Sequence[float], float],
        max_roi_scale: Optional[Union[Sequence[float], float]] = None,
        random_center: bool = True,
        random_size: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(
            keys=keys,
            roi_size=-1,
            max_roi_size=None,
            random_center=random_center,
            random_size=random_size,
            allow_missing_keys=allow_missing_keys,
        )
        self.roi_scale = roi_scale
        self.max_roi_scale = max_roi_scale


@contextlib.contextmanager
def _nullcontext(x):
    """
    This is just like contextlib.nullcontext but also works in Python 3.6.
    """
    yield x


class RandSpatialCropSamplesd(Randomizable, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandSpatialCropSamples`.
    Crop image with random size or specific size ROI to generate a list of N samples.
    It can crop at a random position as center or at the image center. And allows to set
    the minimum size to limit the randomly generated ROI. Suppose all the expected fields
    specified by `keys` have same shape, and add `patch_index` to the corresponding metadata.
    It will return a list of dictionaries for all the cropped images.

    Note: even `random_size=False`, if a dimension of the expected ROI size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped
    results of several images may not have exactly the same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
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
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            used to add `patch_index` to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: When ``num_samples`` is nonpositive.

    """

    backend = RandSpatialCropd.backend

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Union[Sequence[int], int],
        num_samples: int,
        max_roi_size: Optional[Union[Sequence[int], int]] = None,
        random_center: bool = True,
        random_size: bool = True,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self.num_samples = num_samples
        self.cropper = RandSpatialCropd(keys, roi_size, max_roi_size, random_center, random_size, allow_missing_keys)
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSpatialCropSamplesd":
        super().set_random_state(seed, state)
        self.cropper.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        pass

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        ret = []
        for i in range(self.num_samples):
            d = dict(data)
            # deep copy all the unmodified data
            for key in set(data.keys()).difference(set(self.keys)):
                d[key] = deepcopy(data[key])
            cropped = self.cropper(d)
            # self.cropper will have added RandSpatialCropd to the list. Change to RandSpatialCropSamplesd
            for key in self.key_iterator(cropped):
                cropped[self.trace_key(key)][-1][TraceKeys.CLASS_NAME] = self.__class__.__name__  # type: ignore
                cropped[self.trace_key(key)][-1][TraceKeys.ID] = id(self)  # type: ignore
            # add `patch_index` to the metadata
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in cropped:
                    cropped[meta_key] = {}  # type: ignore
                cropped[meta_key][Key.PATCH_INDEX] = i  # type: ignore
            ret.append(cropped)
        return ret

    def inverse(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = deepcopy(dict(data))
        # We changed the transform name from RandSpatialCropd to RandSpatialCropSamplesd
        # Need to revert that since we're calling RandSpatialCropd's inverse
        for key in self.key_iterator(d):
            d[self.trace_key(key)][-1][TraceKeys.CLASS_NAME] = self.cropper.__class__.__name__
            d[self.trace_key(key)][-1][TraceKeys.ID] = id(self.cropper)
        context_manager = allow_missing_keys_mode if self.allow_missing_keys else _nullcontext
        with context_manager(self.cropper):
            return self.cropper.inverse(d)


class CropForegroundd(MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.CropForeground`.
    Crop only the foreground object of the expected images.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    The valid part can be determined by any field in the data with `source_key`, for example:
    - Select values > 0 in image field as the foreground and crop on all fields specified by `keys`.
    - Select label = 3 in label field as the foreground to crop on all fields specified by `keys`.
    - Select label > 0 in the third channel of a One-Hot label field as the foreground to crop all `keys` fields.
    Users can define arbitrary function to select expected foreground from the whole source image or specified
    channels. And it can also add margin to every dim of the bounding box of foreground object.
    """

    backend = CropForeground.backend

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int] = 0,
        allow_smaller: bool = True,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = NumpyPadMode.CONSTANT,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        allow_missing_keys: bool = False,
        **np_kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            source_key: data source to generate the bounding box of foreground, can be image or label, etc.
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            allow_smaller: when computing box size with `margin`, whether allow the image size to be smaller
                than box size, default to `True`. if the margined size is bigger than image size, will pad with
                specified `mode`.
            k_divisible: make each spatial dimension to be divisible by k, default to 1.
                if `k_divisible` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                it also can be a sequence of string, each element corresponds to a key in ``keys``.
            start_coord_key: key to record the start coordinate of spatial bounding box for foreground.
            end_coord_key: key to record the end coordinate of spatial bounding box for foreground.
            allow_missing_keys: don't raise exception if key is missing.
            np_kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
                more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

        """
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.cropper = CropForeground(
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=margin,
            allow_smaller=allow_smaller,
            k_divisible=k_divisible,
            **np_kwargs,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        box_start, box_end = self.cropper.compute_bounding_box(img=d[self.source_key])
        d[self.start_coord_key] = box_start
        d[self.end_coord_key] = box_end
        for key, m in self.key_iterator(d, self.mode):
            self.push_transform(d, key, extra_info={"box_start": box_start, "box_end": box_end})
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            cur_size = np.asarray(d[key].shape[1:])
            extra_info = transform[TraceKeys.EXTRA_INFO]
            box_start = np.asarray(extra_info["box_start"])
            box_end = np.asarray(extra_info["box_end"])
            # first crop the padding part
            roi_start = np.maximum(-box_start, 0)
            roi_end = cur_size - np.maximum(box_end - orig_size, 0)

            d[key] = SpatialCrop(roi_start=roi_start, roi_end=roi_end)(d[key])

            # update bounding box to pad
            pad_to_start = np.maximum(box_start, 0)
            pad_to_end = orig_size - np.minimum(box_end, orig_size)
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            # second pad back the original size
            d[key] = BorderPad(pad)(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandWeightedCropd(Randomizable, MapTransform, InvertibleTransform):
    """
    Samples a list of `num_samples` image patches according to the provided `weight_map`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        w_key: key for the weight map. The corresponding value will be used as the sampling weights,
            it should be a single-channel array in size, for example, `(1, spatial_dim_0, spatial_dim_1, ...)`
        spatial_size: the spatial size of the image patch e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `img` will be used.
        num_samples: number of samples (image patches) to take in the returned list.
        center_coord_key: if specified, the actual sampling location will be stored with the corresponding key.
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            used to add `patch_index` to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_missing_keys: don't raise exception if key is missing.

    See Also:
        :py:class:`monai.transforms.RandWeightedCrop`
    """

    backend = SpatialCrop.backend

    def __init__(
        self,
        keys: KeysCollection,
        w_key: str,
        spatial_size: Union[Sequence[int], int],
        num_samples: int = 1,
        center_coord_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.spatial_size = ensure_tuple(spatial_size)
        self.w_key = w_key
        self.num_samples = int(num_samples)
        self.center_coord_key = center_coord_key
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.centers: List[np.ndarray] = []

    def randomize(self, weight_map: NdarrayOrTensor) -> None:
        self.centers = weighted_patch_samples(
            spatial_size=self.spatial_size, w=weight_map[0], n_samples=self.num_samples, r_state=self.R
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        d = dict(data)
        self.randomize(d[self.w_key])
        _spatial_size = fall_back_tuple(self.spatial_size, d[self.w_key].shape[1:])

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(data) for _ in range(self.num_samples)]
        # fill in the extra keys with unmodified data
        for i in range(self.num_samples):
            for key in set(data.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(data[key])
        for key in self.key_iterator(d):
            img = d[key]
            if img.shape[1:] != d[self.w_key].shape[1:]:
                raise ValueError(
                    f"data {key} and weight map {self.w_key} spatial shape mismatch: "
                    f"{img.shape[1:]} vs {d[self.w_key].shape[1:]}."
                )
            for i, center in enumerate(self.centers):
                cropper = SpatialCrop(roi_center=center, roi_size=_spatial_size)
                orig_size = img.shape[1:]
                results[i][key] = cropper(img)
                self.push_transform(results[i], key, extra_info={"center": center}, orig_size=orig_size)
                if self.center_coord_key:
                    results[i][self.center_coord_key] = center
        # fill in the extra keys with unmodified data
        for i in range(self.num_samples):
            # add `patch_index` to the metadata
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore

        return results

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            current_size = np.asarray(d[key].shape[1:])
            center = transform[TraceKeys.EXTRA_INFO]["center"]
            cropper = SpatialCrop(roi_center=center, roi_size=self.spatial_size)
            # get required pad to start and end
            pad_to_start = np.array([s.indices(o)[0] for s, o in zip(cropper.slices, orig_size)])
            pad_to_end = orig_size - current_size - pad_to_start
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandCropByPosNegLabeld(Randomizable, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    Suppose all the expected fields specified by `keys` have same shape,
    and add `patch_index` to the corresponding metadata.
    And will return a list of dictionaries for all the cropped images.

    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected size,
    and the cropped results of several images may not have exactly the same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center
    to ensure the valid crop ROI.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `data[label_key]` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            used to add `patch_index` to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    backend = RandCropByPosNegLabel.backend

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.centers: Optional[List[List[int]]] = None
        self.allow_smaller = allow_smaller

    def randomize(
        self,
        label: NdarrayOrTensor,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        image: Optional[NdarrayOrTensor] = None,
    ) -> None:
        if fg_indices is None or bg_indices is None:
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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = d.pop(self.fg_indices_key, None) if self.fg_indices_key is not None else None
        bg_indices = d.pop(self.bg_indices_key, None) if self.bg_indices_key is not None else None

        self.randomize(label, fg_indices, bg_indices, image)
        if self.centers is None:
            raise ValueError("no available ROI centers to crop.")

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(self.num_samples)]

        for i, center in enumerate(self.centers):
            # fill in the extra keys with unmodified data
            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(d[key])
            for key in self.key_iterator(d):
                img = d[key]
                orig_size = img.shape[1:]
                roi_size = fall_back_tuple(self.spatial_size, default=orig_size)
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=roi_size)
                results[i][key] = cropper(img)
                self.push_transform(results[i], key, extra_info={"center": center}, orig_size=orig_size)
            # add `patch_index` to the metadata
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore

        return results

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            current_size = np.asarray(d[key].shape[1:])
            center = transform[TraceKeys.EXTRA_INFO]["center"]
            roi_size = fall_back_tuple(self.spatial_size, default=orig_size)
            cropper = SpatialCrop(roi_center=tuple(center), roi_size=roi_size)  # type: ignore
            # get required pad to start and end
            pad_to_start = np.array([s.indices(o)[0] for s, o in zip(cropper.slices, orig_size)])
            pad_to_end = orig_size - current_size - pad_to_start
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandCropByLabelClassesd(Randomizable, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByLabelClasses`.
    Crop random fixed sized regions with the center being a class based on the specified ratios of every class.
    The label data can be One-Hot format array or Argmax data. And will return a list of arrays for all the
    cropped images. For example, crop two (3 x 3) arrays from (5 x 5) array with `ratios=[1, 2, 3, 1]`::

        cropper = RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[3, 3],
            ratios=[1, 2, 3, 1],
            num_classes=4,
            num_samples=2,
        )
        data = {
            "image": np.array([
                [[0.0, 0.3, 0.4, 0.2, 0.0],
                [0.0, 0.1, 0.2, 0.1, 0.4],
                [0.0, 0.3, 0.5, 0.2, 0.0],
                [0.1, 0.2, 0.1, 0.1, 0.0],
                [0.0, 0.1, 0.2, 0.1, 0.0]]
            ]),
            "label": np.array([
                [[0, 0, 0, 0, 0],
                [0, 1, 2, 1, 0],
                [0, 1, 3, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
            ]),
        }
        result = cropper(data)

        The 2 randomly cropped samples of `label` can be:
        [[0, 1, 2],     [[0, 0, 0],
         [0, 1, 3],      [1, 2, 1],
         [0, 0, 0]]      [1, 3, 0]]

    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.
    And if the crop ROI is partly out of the image, will automatically adjust the crop center to ensure the
    valid crop ROI.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding indices of every class.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        ratios: specified ratios of every class in the label to generate crop centers, including background class.
            if None, every class will have the same ratio to generate crop centers.
        num_classes: number of classes for argmax label, not necessary for One-Hot label.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, only return the indices of every class that are within the valid
            region of the image (``image > image_threshold``).
        image_threshold: if enabled `image_key`, use ``image > image_threshold`` to
            determine the valid image content area and select class indices only in this area.
        indices_key: if provided pre-computed indices of every class, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, expect to be 1 dim array
            of spatial indices after flattening. a typical usage is to call `ClassesToIndices` transform first
            and cache the results for better performance.
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            used to add `patch_index` to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will remain
            unchanged.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = RandCropByLabelClasses.backend

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        ratios: Optional[List[Union[float, int]]] = None,
        num_classes: Optional[int] = None,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        indices_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.spatial_size = spatial_size
        self.ratios = ratios
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.indices_key = indices_key
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.centers: Optional[List[List[int]]] = None
        self.allow_smaller = allow_smaller

    def randomize(
        self,
        label: NdarrayOrTensor,
        indices: Optional[List[NdarrayOrTensor]] = None,
        image: Optional[NdarrayOrTensor] = None,
    ) -> None:
        if indices is None:
            indices_ = map_classes_to_indices(label, self.num_classes, image, self.image_threshold)
        else:
            indices_ = indices
        self.centers = generate_label_classes_crop_centers(
            self.spatial_size, self.num_samples, label.shape[1:], indices_, self.ratios, self.R, self.allow_smaller
        )

    def __call__(self, data: Mapping[Hashable, Any]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        indices = d.pop(self.indices_key, None) if self.indices_key is not None else None

        self.randomize(label, indices, image)
        if self.centers is None:
            raise ValueError("no available ROI centers to crop.")

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(self.num_samples)]

        for i, center in enumerate(self.centers):
            # fill in the extra keys with unmodified data
            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(d[key])
            for key in self.key_iterator(d):
                img = d[key]
                orig_size = img.shape[1:]
                roi_size = fall_back_tuple(self.spatial_size, default=orig_size)
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=roi_size)
                results[i][key] = cropper(img)
                self.push_transform(results[i], key, extra_info={"center": center}, orig_size=orig_size)
            # add `patch_index` to the metadata
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore

        return results

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            current_size = np.asarray(d[key].shape[1:])
            center = transform[TraceKeys.EXTRA_INFO]["center"]
            roi_size = fall_back_tuple(self.spatial_size, default=orig_size)
            cropper = SpatialCrop(roi_center=tuple(center), roi_size=roi_size)  # type: ignore
            # get required pad to start and end
            pad_to_start = np.array([s.indices(o)[0] for s, o in zip(cropper.slices, orig_size)])
            pad_to_end = orig_size - current_size - pad_to_start
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class ResizeWithPadOrCropd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ResizeWithPadOrCrop`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        spatial_size: the spatial size of output data after padding or crop.
            If has non-positive values, the corresponding size of input image will be used (no padding).
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function for padding. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
        method: {``"symmetric"``, ``"end"``}
            Pad image symmetrically on every side or only pad at the end sides. Defaults to ``"symmetric"``.
        np_kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
            more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

    """

    backend = ResizeWithPadOrCrop.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT,
        allow_missing_keys: bool = False,
        method: Union[Method, str] = Method.SYMMETRIC,
        **np_kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padcropper = ResizeWithPadOrCrop(spatial_size=spatial_size, method=method, **np_kwargs)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, m in self.key_iterator(d, self.mode):
            orig_size = d[key].shape[1:]
            d[key] = self.padcropper(d[key], mode=m)
            self.push_transform(d, key, orig_size=orig_size, extra_info={"mode": m.value if isinstance(m, Enum) else m})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.array(transform[TraceKeys.ORIG_SIZE])
            current_size = np.array(d[key].shape[1:])
            # Unfortunately, we can't just use ResizeWithPadOrCrop with original size because of odd/even rounding.
            # Instead, we first pad any smaller dimensions, and then we crop any larger dimensions.

            # First, do pad
            if np.any((orig_size - current_size) > 0):
                pad_to_start = np.floor((orig_size - current_size) / 2).astype(int)
                # in each direction, if original size is even and current size is odd, += 1
                pad_to_start[np.logical_and(orig_size % 2 == 0, current_size % 2 == 1)] += 1
                pad_to_start[pad_to_start < 0] = 0
                pad_to_end = orig_size - current_size - pad_to_start
                pad_to_end[pad_to_end < 0] = 0
                pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
                d[key] = BorderPad(pad)(d[key])

            # Next crop
            if np.any((orig_size - current_size) < 0):
                if self.padcropper.padder.method == Method.SYMMETRIC:
                    roi_center = [floor(i / 2) if r % 2 == 0 else (i - 1) // 2 for r, i in zip(orig_size, current_size)]
                else:
                    roi_center = [floor(r / 2) if r % 2 == 0 else (r - 1) // 2 for r in orig_size]

                d[key] = SpatialCrop(roi_center, orig_size)(d[key])

            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class BoundingRectd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.BoundingRect`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        bbox_key_postfix: the output bounding box coordinates will be
            written to the value of `{key}_{bbox_key_postfix}`.
        select_fn: function to select expected foreground, default is to select values > 0.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = BoundingRect.backend

    def __init__(
        self,
        keys: KeysCollection,
        bbox_key_postfix: str = "bbox",
        select_fn: Callable = is_positive,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.bbox = BoundingRect(select_fn=select_fn)
        self.bbox_key_postfix = bbox_key_postfix

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        """
        See also: :py:class:`monai.transforms.utils.generate_spatial_bounding_box`.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            bbox = self.bbox(d[key])
            key_to_add = f"{key}_{self.bbox_key_postfix}"
            if key_to_add in d:
                raise KeyError(f"Bounding box data with key {key_to_add} already exists.")
            d[key_to_add] = bbox
        return d


PadBaseD = PadBaseDict = PadBased
SpatialPadD = SpatialPadDict = SpatialPadd
BorderPadD = BorderPadDict = BorderPadd
DivisiblePadD = DivisiblePadDict = DivisiblePadd
SpatialCropD = SpatialCropDict = SpatialCropd
CenterSpatialCropD = CenterSpatialCropDict = CenterSpatialCropd
CenterScaleCropD = CenterScaleCropDict = CenterScaleCropd
RandSpatialCropD = RandSpatialCropDict = RandSpatialCropd
RandScaleCropD = RandScaleCropDict = RandScaleCropd
RandSpatialCropSamplesD = RandSpatialCropSamplesDict = RandSpatialCropSamplesd
CropForegroundD = CropForegroundDict = CropForegroundd
RandWeightedCropD = RandWeightedCropDict = RandWeightedCropd
RandCropByPosNegLabelD = RandCropByPosNegLabelDict = RandCropByPosNegLabeld
RandCropByLabelClassesD = RandCropByLabelClassesDict = RandCropByLabelClassesd
ResizeWithPadOrCropD = ResizeWithPadOrCropDict = ResizeWithPadOrCropd
BoundingRectD = BoundingRectDict = BoundingRectd
