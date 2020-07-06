# Copyright 2020 MONAI Consortium
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

from typing import Callable, Optional, Sequence, Union

from monai.config import IndexSelection, KeysCollection
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.croppad.array import CenterSpatialCrop, DivisiblePad, SpatialCrop, SpatialPad, BorderPad
from monai.transforms.utils import generate_pos_neg_label_crop_centers, generate_spatial_bounding_box
from monai.utils import ensure_tuple, ensure_tuple_rep, fall_back_tuple, NumpyPadMode, Method

NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]


class SpatialPadd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialPad`.
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size,
        method: Union[Method, str] = Method.SYMMETRIC,
        mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            spatial_size (list): the spatial size of output data after padding.
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

    def __call__(self, data):
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
        self, keys: KeysCollection, spatial_border, mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            spatial_border (int or sequence of int): specified size for every spatial border. it can be 3 shapes:

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

    def __call__(self, data):
        d = dict(data)
        for key, m in zip(self.keys, self.mode):
            d[key] = self.padder(d[key], mode=m)
        return d


class DivisiblePadd(MapTransform):
    """
    Pad the input data, so that the spatial sizes are divisible by `k`.
    Dictionary-based wrapper of :py:class:`monai.transforms.DivisiblePad`.
    """

    def __init__(self, keys: KeysCollection, k, mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            k (int or sequence of int): the target k for each spatial dimension.
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

    def __call__(self, data):
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

    def __init__(self, keys: KeysCollection, roi_center=None, roi_size=None, roi_start=None, roi_end=None):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            roi_center (list or tuple): voxel coordinates for center of the crop ROI.
            roi_size (list or tuple): size of the crop ROI.
            roi_start (list or tuple): voxel coordinates for start of the crop ROI.
            roi_end (list or tuple): voxel coordinates for end of the crop ROI.
        """
        super().__init__(keys)
        self.cropper = SpatialCrop(roi_center, roi_size, roi_start, roi_end)

    def __call__(self, data):
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
        roi_size (list, tuple): the size of the crop region e.g. [224,224,128]
            If its components have non-positive values, the corresponding size of input image will be used.
    """

    def __init__(self, keys: KeysCollection, roi_size):
        super().__init__(keys)
        self.cropper = CenterSpatialCrop(roi_size)

    def __call__(self, data):
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
        roi_size (list, tuple): if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            If its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.
    """

    def __init__(self, keys: KeysCollection, roi_size, random_center: bool = True, random_size: bool = True):
        super().__init__(keys)
        self.roi_size = roi_size
        self.random_center = random_center
        self.random_size = random_size
        self._slices = None
        self._size = None

    def randomize(self, img_size):
        self._size = fall_back_tuple(self.roi_size, img_size)
        if self.random_size:
            self._size = [self.R.randint(low=self._size[i], high=img_size[i] + 1) for i in range(len(img_size))]
        if self.random_center:
            valid_size = get_valid_patch_size(img_size, self._size)
            self._slices = (slice(None),) + get_random_patch(img_size, valid_size, self.R)

    def __call__(self, data):
        d = dict(data)
        self.randomize(d[self.keys[0]].shape[1:])  # image shape from the first data key
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
        roi_size (list, tuple): if `random_size` is True, the spatial size of the minimum crop region.
            if `random_size` is False, specify the expected ROI size to crop. e.g. [224, 224, 128]
        num_samples: number of samples (crop regions) to take in the returned list.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.
    """

    def __init__(
        self, keys: KeysCollection, roi_size, num_samples: int, random_center: bool = True, random_size: bool = True
    ):
        super().__init__(keys)
        if num_samples < 1:
            raise ValueError("number of samples must be greater than 0.")
        self.num_samples = num_samples
        self.cropper = RandSpatialCropd(keys, roi_size, random_center, random_size)

    def randomize(self):
        pass

    def __call__(self, data):
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
        channel_indexes: Optional[IndexSelection] = None,
        margin: int = 0,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            source_key: data source to generate the bounding box of foreground, can be image or label, etc.
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indexes: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin to all dims of the bounding box.
        """
        super().__init__(keys)
        self.source_key = source_key
        self.select_fn = select_fn
        self.channel_indexes = ensure_tuple(channel_indexes) if channel_indexes is not None else None
        self.margin = margin

    def __call__(self, data):
        d = dict(data)
        box_start, box_end = generate_spatial_bounding_box(
            d[self.source_key], self.select_fn, self.channel_indexes, self.margin
        )
        cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)
        for key in self.keys:
            d[key] = cropper(d[key])
        return d


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
        spatial_size (sequence of int): the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
        pos: used to calculate the ratio ``pos / (pos + neg)`` for the probability to pick a
            foreground voxel as a center rather than a background voxel.
        neg: used to calculate the ratio ``pos / (pos + neg)`` for the probability to pick a
            foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.

    Raises:
        ValueError: pos and neg must be greater than or equal to 0.
        ValueError: pos and neg cannot both be 0.

    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
    ):
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError("pos and neg must be greater than or equal to 0.")
        if pos + neg == 0:
            raise ValueError("pos and neg cannot both be 0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.centers = None

    def randomize(self, label, image):
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        self.centers = generate_pos_neg_label_crop_centers(
            label, self.spatial_size, self.num_samples, self.pos_ratio, image, self.image_threshold, self.R
        )

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        self.randomize(label, image)
        results = [dict() for _ in range(self.num_samples)]
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


SpatialPadD = SpatialPadDict = SpatialPadd
BorderPadD = BorderPadDict = BorderPadd
DivisiblePadD = DivisiblePadDict = DivisiblePadd
SpatialCropD = SpatialCropDict = SpatialCropd
CenterSpatialCropD = CenterSpatialCropDict = CenterSpatialCropd
RandSpatialCropD = RandSpatialCropDict = RandSpatialCropd
RandSpatialCropSamplesD = RandSpatialCropSamplesDict = RandSpatialCropSamplesd
CropForegroundD = CropForegroundDict = CropForegroundd
RandCropByPosNegLabelD = RandCropByPosNegLabelDict = RandCropByPosNegLabeld
