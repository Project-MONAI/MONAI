# Copyright 2020 - 2021 MONAI Consortium
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

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from monai.config import IndexSelection, KeysCollection
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.croppad.array import (
    BorderPad,
    BoundingRect,
    CenterSpatialCrop,
    DivisiblePad,
    ResizeWithPadOrCrop,
    SpatialCrop,
    SpatialPad,
)
from monai.transforms.utils import (
    generate_pos_neg_label_crop_centers,
    generate_spatial_bounding_box,
    map_binary_to_indices,
    weighted_patch_samples,
)
from monai.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple

__all__ = [
    "NumpyPadModeSequence",
    "SpatialPadd",
    "BorderPadd",
    "DivisiblePadd",
    "SpatialCropd",
    "CenterSpatialCropd",
    "RandSpatialCropd",
    "RandSpatialCropSamplesd",
    "CropForegroundd",
    "RandWeightedCropd",
    "RandCropByPosNegLabeld",
    "ResizeWithPadOrCropd",
    "BoundingRectd",
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
]

NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]


class SpatialPadd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialPad`.
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        method: Union[Method, str] = Method.SYMMETRIC,
        mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            spatial_size: the spatial size of output data after padding.
                If its components have non-positive values, the corresponding size of input image will be used.
            method: {``"symmetric"``, ``"end"``}
                Pad image symmetric on every side or only pad at the end sides. Defaults to ``"symmetric"``.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.

        """
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padder = SpatialPad(spatial_size, method)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key, m in zip(self.keys, self.mode):
            d[key] = self.padder(d[key], mode=m)
        return d


class BorderPadd(MapTransform):
    """
    Pad the input data by adding specified borders to every dimension.
    Dictionary-based wrapper of :py:class:`monai.transforms.BorderPad`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_border: Union[Sequence[int], int],
        mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT,
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

            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.

        """
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padder = BorderPad(spatial_border=spatial_border)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key, m in zip(self.keys, self.mode):
            d[key] = self.padder(d[key], mode=m)
        return d


class DivisiblePadd(MapTransform):
    """
    Pad the input data, so that the spatial sizes are divisible by `k`.
    Dictionary-based wrapper of :py:class:`monai.transforms.DivisiblePad`.
    """

    def __init__(
        self, keys: KeysCollection, k: Union[Sequence[int], int], mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            k: the target k for each spatial dimension.
                if `k` is negative or 0, the original size is preserved.
                if `k` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.

        See also :py:class:`monai.transforms.SpatialPad`

        """
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padder = DivisiblePad(k=k)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key, m in zip(self.keys, self.mode):
            d[key] = self.padder(d[key], mode=m)
        return d


class SpatialCropd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialCrop`.
    Either a spatial center and size must be provided, or alternatively if center and size
    are not provided, the start and end coordinates of the ROI must be provided.
    """

    def __init__(
        self,
        keys: KeysCollection,
        roi_center: Optional[Sequence[int]] = None,
        roi_size: Optional[Sequence[int]] = None,
        roi_start: Optional[Sequence[int]] = None,
        roi_end: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI.
        """
        super().__init__(keys)
        self.cropper = SpatialCrop(roi_center, roi_size, roi_start, roi_end)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.cropper(d[key])
        return d


class CenterSpatialCropd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CenterSpatialCrop`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: the size of the crop region e.g. [224,224,128]
            If its components have non-positive values, the corresponding size of input image will be used.
    """

    def __init__(self, keys: KeysCollection, roi_size: Union[Sequence[int], int]) -> None:
        super().__init__(keys)
        self.cropper = CenterSpatialCrop(roi_size)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.cropper(d[key])
        return d


class RandSpatialCropd(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandSpatialCrop`.
    Crop image with random size or specific size ROI. It can crop at a random position as
    center or at the image center. And allows to set the minimum size to limit the randomly
    generated ROI. Suppose all the expected fields specified by `keys` have same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            If its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Union[Sequence[int], int],
        random_center: bool = True,
        random_size: bool = True,
    ) -> None:
        super().__init__(keys)
        self.roi_size = roi_size
        self.random_center = random_center
        self.random_size = random_size
        self._slices: Optional[Tuple[slice, ...]] = None
        self._size: Optional[Sequence[int]] = None

    def randomize(self, img_size: Sequence[int]) -> None:
        self._size = fall_back_tuple(self.roi_size, img_size)
        if self.random_size:
            self._size = [self.R.randint(low=self._size[i], high=img_size[i] + 1) for i in range(len(img_size))]
        if self.random_center:
            valid_size = get_valid_patch_size(img_size, self._size)
            self._slices = (slice(None),) + get_random_patch(img_size, valid_size, self.R)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize(d[self.keys[0]].shape[1:])  # image shape from the first data key
        if self._size is None:
            raise AssertionError
        for key in self.keys:
            if self.random_center:
                d[key] = d[key][self._slices]
            else:
                cropper = CenterSpatialCrop(self._size)
                d[key] = cropper(d[key])
        return d


class RandSpatialCropSamplesd(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandSpatialCropSamples`.
    Crop image with random size or specific size ROI to generate a list of N samples.
    It can crop at a random position as center or at the image center. And allows to set
    the minimum size to limit the randomly generated ROI. Suppose all the expected fields
    specified by `keys` have same shape.
    It will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: if `random_size` is True, the spatial size of the minimum crop region.
            if `random_size` is False, specify the expected ROI size to crop. e.g. [224, 224, 128]
        num_samples: number of samples (crop regions) to take in the returned list.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.

    Raises:
        ValueError: When ``num_samples`` is nonpositive.

    """

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Union[Sequence[int], int],
        num_samples: int,
        random_center: bool = True,
        random_size: bool = True,
    ) -> None:
        super().__init__(keys)
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self.num_samples = num_samples
        self.cropper = RandSpatialCropd(keys, roi_size, random_center, random_size)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Randomizable":
        super().set_random_state(seed=seed, state=state)
        self.cropper.set_random_state(state=self.R)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        pass

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> List[Dict[Hashable, np.ndarray]]:
        return [self.cropper(data) for _ in range(self.num_samples)]


class CropForegroundd(MapTransform):
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

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        select_fn: Callable = lambda x: x > 0,
        channel_indices: Optional[IndexSelection] = None,
        margin: int = 0,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
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
            start_coord_key: key to record the start coordinate of spatial bounding box for foreground.
            end_coord_key: key to record the end coordinate of spatial bounding box for foreground.
        """
        super().__init__(keys)
        self.source_key = source_key
        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        box_start, box_end = generate_spatial_bounding_box(
            d[self.source_key], self.select_fn, self.channel_indices, self.margin
        )
        d[self.start_coord_key] = box_start
        d[self.end_coord_key] = box_end
        cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)
        for key in self.keys:
            d[key] = cropper(d[key])
        return d


class RandWeightedCropd(Randomizable, MapTransform):
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

    See Also:
        :py:class:`monai.transforms.RandWeightedCrop`
    """

    def __init__(
        self,
        keys: KeysCollection,
        w_key: str,
        spatial_size: Union[Sequence[int], int],
        num_samples: int = 1,
        center_coord_key: Optional[str] = None,
    ):
        super().__init__(keys)
        self.spatial_size = ensure_tuple(spatial_size)
        self.w_key = w_key
        self.num_samples = int(num_samples)
        self.center_coord_key = center_coord_key
        self.centers: List[np.ndarray] = []

    def randomize(self, weight_map: np.ndarray) -> None:
        self.centers = weighted_patch_samples(
            spatial_size=self.spatial_size, w=weight_map[0], n_samples=self.num_samples, r_state=self.R
        )

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> List[Dict[Hashable, np.ndarray]]:
        d = dict(data)
        self.randomize(d[self.w_key])
        _spatial_size = fall_back_tuple(self.spatial_size, d[self.w_key].shape[1:])

        results: List[Dict[Hashable, np.ndarray]] = [{} for _ in range(self.num_samples)]
        for key in data.keys():
            if key in self.keys:
                img = d[key]
                if img.shape[1:] != d[self.w_key].shape[1:]:
                    raise ValueError(
                        f"data {key} and weight map {self.w_key} spatial shape mismatch: "
                        f"{img.shape[1:]} vs {d[self.w_key].shape[1:]}."
                    )
                for i, center in enumerate(self.centers):
                    cropper = SpatialCrop(roi_center=center, roi_size=_spatial_size)
                    results[i][key] = cropper(img)
                    if self.center_coord_key:
                        results[i][self.center_coord_key] = center
            else:
                for i in range(self.num_samples):
                    results[i][key] = data[key]

        return results


class RandCropByPosNegLabeld(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
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

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

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
    ) -> None:
        super().__init__(keys)
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
        self.centers: Optional[List[List[np.ndarray]]] = None

    def randomize(
        self,
        label: np.ndarray,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_, bg_indices_, self.R
        )

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> List[Dict[Hashable, np.ndarray]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = d.get(self.fg_indices_key, None) if self.fg_indices_key is not None else None
        bg_indices = d.get(self.bg_indices_key, None) if self.bg_indices_key is not None else None

        self.randomize(label, fg_indices, bg_indices, image)
        if not isinstance(self.spatial_size, tuple):
            raise AssertionError
        if self.centers is None:
            raise AssertionError
        results: List[Dict[Hashable, np.ndarray]] = [{} for _ in range(self.num_samples)]
        for key in data.keys():
            if key in self.keys:
                img = d[key]
                for i, center in enumerate(self.centers):
                    cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)
                    results[i][key] = cropper(img)
            else:
                for i in range(self.num_samples):
                    results[i][key] = data[key]

        return results


class ResizeWithPadOrCropd(MapTransform):
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

    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        mode: Union[NumpyPadMode, str] = NumpyPadMode.CONSTANT,
    ) -> None:
        super().__init__(keys)
        self.padcropper = ResizeWithPadOrCrop(spatial_size=spatial_size, mode=mode)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.padcropper(d[key])
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
    """

    def __init__(self, keys: KeysCollection, bbox_key_postfix: str = "bbox", select_fn: Callable = lambda x: x > 0):
        super().__init__(keys=keys)
        self.bbox = BoundingRect(select_fn=select_fn)
        self.bbox_key_postfix = bbox_key_postfix

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        """
        See also: :py:class:`monai.transforms.utils.generate_spatial_bounding_box`.
        """
        d = dict(data)
        for key in self.keys:
            bbox = self.bbox(d[key])
            key_to_add = f"{key}_{self.bbox_key_postfix}"
            if key_to_add in d:
                raise KeyError(f"Bounding box data with key {key_to_add} already exists.")
            d[key_to_add] = bbox
        return d


SpatialPadD = SpatialPadDict = SpatialPadd
BorderPadD = BorderPadDict = BorderPadd
DivisiblePadD = DivisiblePadDict = DivisiblePadd
SpatialCropD = SpatialCropDict = SpatialCropd
CenterSpatialCropD = CenterSpatialCropDict = CenterSpatialCropd
RandSpatialCropD = RandSpatialCropDict = RandSpatialCropd
RandSpatialCropSamplesD = RandSpatialCropSamplesDict = RandSpatialCropSamplesd
CropForegroundD = CropForegroundDict = CropForegroundd
RandWeightedCropD = RandWeightedCropDict = RandWeightedCropd
RandCropByPosNegLabelD = RandCropByPosNegLabelDict = RandCropByPosNegLabeld
ResizeWithPadOrCropD = ResizeWithPadOrCropDict = ResizeWithPadOrCropd
BoundingRectD = BoundingRectDict = BoundingRectd
