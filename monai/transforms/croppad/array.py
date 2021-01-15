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
A collection of "vanilla" transforms for crop and pad operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from monai.config import IndexSelection
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.compose import Randomizable, Transform
from monai.transforms.utils import (
    generate_pos_neg_label_crop_centers,
    generate_spatial_bounding_box,
    map_binary_to_indices,
    weighted_patch_samples,
)
from monai.utils import Method, NumpyPadMode, ensure_tuple, fall_back_tuple

__all__ = [
    "SpatialPad",
    "BorderPad",
    "DivisiblePad",
    "SpatialCrop",
    "CenterSpatialCrop",
    "RandSpatialCrop",
    "RandSpatialCropSamples",
    "CropForeground",
    "RandWeightedCrop",
    "RandCropByPosNegLabel",
    "ResizeWithPadOrCrop",
    "BoundingRect",
]


class SpatialPad(Transform):
    """
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.
    Uses np.pad so in practice, a mode needs to be provided. See numpy.lib.arraypad.pad
    for additional details.

    Args:
        spatial_size: the spatial size of output data after padding.
            If its components have non-positive values, the corresponding size of input image will be used (no padding).
        method: {``"symmetric"``, ``"end"``}
            Pad image symmetric on every side or only pad at the end sides. Defaults to ``"symmetric"``.
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        method: Union[Method, str] = Method.SYMMETRIC,
        mode: Union[NumpyPadMode, str] = NumpyPadMode.CONSTANT,
    ) -> None:
        self.spatial_size = spatial_size
        self.method: Method = Method(method)
        self.mode: NumpyPadMode = NumpyPadMode(mode)

    def _determine_data_pad_width(self, data_shape: Sequence[int]) -> List[Tuple[int, int]]:
        self.spatial_size = fall_back_tuple(self.spatial_size, data_shape)
        if self.method == Method.SYMMETRIC:
            pad_width = []
            for i in range(len(self.spatial_size)):
                width = max(self.spatial_size[i] - data_shape[i], 0)
                pad_width.append((width // 2, width - (width // 2)))
            return pad_width
        return [(0, max(self.spatial_size[i] - data_shape[i], 0)) for i in range(len(self.spatial_size))]

    def __call__(self, img: np.ndarray, mode: Optional[Union[NumpyPadMode, str]] = None) -> np.ndarray:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``self.mode``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        """
        data_pad_width = self._determine_data_pad_width(img.shape[1:])
        all_pad_width = [(0, 0)] + data_pad_width
        if not np.asarray(all_pad_width).any():
            # all zeros, skip padding
            return img
        img = np.pad(img, all_pad_width, mode=self.mode.value if mode is None else NumpyPadMode(mode).value)
        return img


class BorderPad(Transform):
    """
    Pad the input data by adding specified borders to every dimension.

    Args:
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
    """

    def __init__(
        self, spatial_border: Union[Sequence[int], int], mode: Union[NumpyPadMode, str] = NumpyPadMode.CONSTANT
    ) -> None:
        self.spatial_border = spatial_border
        self.mode: NumpyPadMode = NumpyPadMode(mode)

    def __call__(self, img: np.ndarray, mode: Optional[Union[NumpyPadMode, str]] = None) -> np.ndarray:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``self.mode``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

        Raises:
            ValueError: When ``self.spatial_border`` contains a nonnegative int.
            ValueError: When ``self.spatial_border`` length is not one of
                [1, len(spatial_shape), 2*len(spatial_shape)].

        """
        spatial_shape = img.shape[1:]
        spatial_border = ensure_tuple(self.spatial_border)
        for b in spatial_border:
            if not isinstance(b, int) or b < 0:
                raise ValueError(f"self.spatial_border must contain only nonnegative ints, got {spatial_border}.")

        if len(spatial_border) == 1:
            data_pad_width = [(spatial_border[0], spatial_border[0]) for _ in range(len(spatial_shape))]
        elif len(spatial_border) == len(spatial_shape):
            data_pad_width = [(spatial_border[i], spatial_border[i]) for i in range(len(spatial_shape))]
        elif len(spatial_border) == len(spatial_shape) * 2:
            data_pad_width = [(spatial_border[2 * i], spatial_border[2 * i + 1]) for i in range(len(spatial_shape))]
        else:
            raise ValueError(
                f"Unsupported spatial_border length: {len(spatial_border)}, available options are "
                f"[1, len(spatial_shape)={len(spatial_shape)}, 2*len(spatial_shape)={2*len(spatial_shape)}]."
            )

        return np.pad(
            img, [(0, 0)] + data_pad_width, mode=self.mode.value if mode is None else NumpyPadMode(mode).value
        )


class DivisiblePad(Transform):
    """
    Pad the input data, so that the spatial sizes are divisible by `k`.
    """

    def __init__(self, k: Union[Sequence[int], int], mode: Union[NumpyPadMode, str] = NumpyPadMode.CONSTANT) -> None:
        """
        Args:
            k: the target k for each spatial dimension.
                if `k` is negative or 0, the original size is preserved.
                if `k` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

        See also :py:class:`monai.transforms.SpatialPad`
        """
        self.k = k
        self.mode: NumpyPadMode = NumpyPadMode(mode)

    def __call__(self, img: np.ndarray, mode: Optional[Union[NumpyPadMode, str]] = None) -> np.ndarray:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first
                and padding doesn't apply to the channel dim.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``self.mode``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        """
        spatial_shape = img.shape[1:]
        k = fall_back_tuple(self.k, (1,) * len(spatial_shape))
        new_size = []
        for k_d, dim in zip(k, spatial_shape):
            new_dim = int(np.ceil(dim / k_d) * k_d) if k_d > 0 else dim
            new_size.append(new_dim)

        spatial_pad = SpatialPad(spatial_size=new_size, method=Method.SYMMETRIC, mode=mode or self.mode)
        return spatial_pad(img)


class SpatialCrop(Transform):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    It can support to crop ND spatial (channel-first) data.
    Either a spatial center and size must be provided, or alternatively,
    if center and size are not provided, the start and end coordinates of the ROI must be provided.
    """

    def __init__(
        self,
        roi_center: Optional[Sequence[int]] = None,
        roi_size: Optional[Sequence[int]] = None,
        roi_start: Optional[Sequence[int]] = None,
        roi_end: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI.
        """
        if roi_center is not None and roi_size is not None:
            roi_center = np.asarray(roi_center, dtype=np.int16)
            roi_size = np.asarray(roi_size, dtype=np.int16)
            self.roi_start = np.maximum(roi_center - np.floor_divide(roi_size, 2), 0)
            self.roi_end = np.maximum(self.roi_start + roi_size, self.roi_start)
        else:
            if roi_start is None or roi_end is None:
                raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
            self.roi_start = np.maximum(np.asarray(roi_start, dtype=np.int16), 0)
            self.roi_end = np.maximum(np.asarray(roi_end, dtype=np.int16), self.roi_start)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        sd = min(len(self.roi_start), len(self.roi_end), len(img.shape[1:]))  # spatial dims
        slices = [slice(None)] + [slice(s, e) for s, e in zip(self.roi_start[:sd], self.roi_end[:sd])]
        return img[tuple(slices)]


class CenterSpatialCrop(Transform):
    """
    Crop at the center of image with specified ROI size.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            If its components have non-positive values, the corresponding size of input image will be used.
    """

    def __init__(self, roi_size: Union[Sequence[int], int]) -> None:
        self.roi_size = roi_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        self.roi_size = fall_back_tuple(self.roi_size, img.shape[1:])
        center = [i // 2 for i in img.shape[1:]]
        cropper = SpatialCrop(roi_center=center, roi_size=self.roi_size)
        return cropper(img)


class RandSpatialCrop(Randomizable, Transform):
    """
    Crop image with random size or specific size ROI. It can crop at a random position as center
    or at the image center. And allows to set the minimum size to limit the randomly generated ROI.

    Args:
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            If its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.
    """

    def __init__(
        self, roi_size: Union[Sequence[int], int], random_center: bool = True, random_size: bool = True
    ) -> None:
        self.roi_size = roi_size
        self.random_center = random_center
        self.random_size = random_size
        self._size: Optional[Sequence[int]] = None
        self._slices: Optional[Tuple[slice, ...]] = None

    def randomize(self, img_size: Sequence[int]) -> None:
        self._size = fall_back_tuple(self.roi_size, img_size)
        if self.random_size:
            self._size = tuple((self.R.randint(low=self._size[i], high=img_size[i] + 1) for i in range(len(img_size))))
        if self.random_center:
            valid_size = get_valid_patch_size(img_size, self._size)
            self._slices = (slice(None),) + get_random_patch(img_size, valid_size, self.R)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        self.randomize(img.shape[1:])
        if self._size is None:
            raise AssertionError
        if self.random_center:
            return img[self._slices]
        cropper = CenterSpatialCrop(self._size)
        return cropper(img)


class RandSpatialCropSamples(Randomizable, Transform):
    """
    Crop image with random size or specific size ROI to generate a list of N samples.
    It can crop at a random position as center or at the image center. And allows to set
    the minimum size to limit the randomly generated ROI.
    It will return a list of cropped images.

    Args:
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
        roi_size: Union[Sequence[int], int],
        num_samples: int,
        random_center: bool = True,
        random_size: bool = True,
    ) -> None:
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self.num_samples = num_samples
        self.cropper = RandSpatialCrop(roi_size, random_center, random_size)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Randomizable":
        super().set_random_state(seed=seed, state=state)
        self.cropper.set_random_state(state=self.R)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        pass

    def __call__(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        cropping doesn't change the channel dim.
        """
        return [self.cropper(img) for _ in range(self.num_samples)]


class CropForeground(Transform):
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
        cropper = CropForeground(select_fn=lambda x: x > 1, margin=0)
        print(cropper(image))
        [[[2, 1],
          [3, 2],
          [2, 1]]]

    """

    def __init__(
        self,
        select_fn: Callable = lambda x: x > 0,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int] = 0,
        return_coords: bool = False,
    ) -> None:
        """
        Args:
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            return_coords: whether return the coordinates of spatial bounding box for foreground.
        """
        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin
        self.return_coords = return_coords

    def __call__(self, img: np.ndarray):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        box_start, box_end = generate_spatial_bounding_box(img, self.select_fn, self.channel_indices, self.margin)
        cropped = SpatialCrop(roi_start=box_start, roi_end=box_end)(img)

        if self.return_coords:
            return cropped, box_start, box_end
        return cropped


class RandWeightedCrop(Randomizable, Transform):
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

    def __init__(
        self, spatial_size: Union[Sequence[int], int], num_samples: int = 1, weight_map: Optional[np.ndarray] = None
    ):
        self.spatial_size = ensure_tuple(spatial_size)
        self.num_samples = int(num_samples)
        self.weight_map = weight_map
        self.centers: List[np.ndarray] = []

    def randomize(self, weight_map: np.ndarray) -> None:
        self.centers = weighted_patch_samples(
            spatial_size=self.spatial_size, w=weight_map[0], n_samples=self.num_samples, r_state=self.R
        )  # using only the first channel as weight map

    def __call__(self, img: np.ndarray, weight_map: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Args:
            img: input image to sample patches from. assuming `img` is a channel-first array.
            weight_map: weight map used to generate patch samples. The weights must be non-negative.
                Each element denotes a sampling weight of the spatial location. 0 indicates no sampling.
                It should be a single-channel array in shape, for example, `(1, spatial_dim_0, spatial_dim_1, ...)`

        Returns:
            A list of image patches
        """
        if weight_map is None:
            weight_map = self.weight_map
        if weight_map is None:
            raise ValueError("weight map must be provided for weighted patch sampling.")
        if img.shape[1:] != weight_map.shape[1:]:
            raise ValueError(f"image and weight map spatial shape mismatch: {img.shape[1:]} vs {weight_map.shape[1:]}.")
        self.randomize(weight_map)
        _spatial_size = fall_back_tuple(self.spatial_size, weight_map.shape[1:])
        results = []
        for center in self.centers:
            cropper = SpatialCrop(roi_center=center, roi_size=_spatial_size)
            results.append(cropper(img))
        return results


class RandCropByPosNegLabel(Randomizable, Transform):
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

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `label` will be used.
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

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        label: Optional[np.ndarray] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: Optional[np.ndarray] = None,
        image_threshold: float = 0.0,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = ensure_tuple(spatial_size)
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.fg_indices = fg_indices
        self.bg_indices = bg_indices

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

    def __call__(
        self,
        img: np.ndarray,
        label: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
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

        """
        if label is None:
            label = self.label
        if image is None:
            image = self.image
        if fg_indices is None or bg_indices is None:
            if self.fg_indices is not None and self.bg_indices is not None:
                fg_indices = self.fg_indices
                bg_indices = self.bg_indices
            else:
                fg_indices, bg_indices = map_binary_to_indices(label, image, self.image_threshold)
        self.randomize(label, fg_indices, bg_indices, image)
        results: List[np.ndarray] = []
        if self.centers is not None:
            for center in self.centers:
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)
                results.append(cropper(img))

        return results


class ResizeWithPadOrCrop(Transform):
    """
    Resize an image to a target spatial size by either centrally cropping the image or
    padding it evenly with a user-specified mode.
    When the dimension is smaller than the target size, do symmetric padding along that dim.
    When the dimension is larger than the target size, do central cropping along that dim.

    Args:
        spatial_size: the spatial size of output data after padding or crop.
            If has non-positive values, the corresponding size of input image will be used (no padding).
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function for padding. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        mode: Union[NumpyPadMode, str] = NumpyPadMode.CONSTANT,
    ):
        self.padder = SpatialPad(spatial_size=spatial_size, mode=mode)
        self.cropper = CenterSpatialCrop(roi_size=spatial_size)

    def __call__(self, img: np.ndarray, mode: Optional[Union[NumpyPadMode, str]] = None) -> np.ndarray:
        """
        Args:
            img: data to pad or crop, assuming `img` is channel-first and
                padding or cropping doesn't apply to the channel dim.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function for padding.
                If None, defaults to the ``mode`` in construction.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        """
        return self.padder(self.cropper(img), mode=mode)


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
    This function returns [-1, -1, ...] if there's no positive intensity.

    Args:
        select_fn: function to select expected foreground, default is to select values > 0.
    """

    def __init__(self, select_fn: Callable = lambda x: x > 0) -> None:
        self.select_fn = select_fn

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        See also: :py:class:`monai.transforms.utils.generate_spatial_bounding_box`.
        """
        bbox = []

        for channel in range(img.shape[0]):
            start_, end_ = generate_spatial_bounding_box(img, select_fn=self.select_fn, channel_indices=channel)
            bbox.append([i for k in zip(start_, end_) for i in k])

        return np.stack(bbox, axis=0)
