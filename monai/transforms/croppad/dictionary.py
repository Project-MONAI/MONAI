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

from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.croppad.array import SpatialCrop, CenterSpatialCrop, SpatialPad
from monai.transforms.utils import generate_pos_neg_label_crop_centers, generate_spatial_bounding_box
from monai.utils.misc import ensure_tuple, ensure_tuple_rep


class SpatialPadd(MapTransform):
    """
    dictionary-based wrapper of :py:class:`monai.transforms.SpatialPad`.
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.
    """

    def __init__(self, keys, spatial_size, method="symmetric", mode="constant"):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            spatial_size (list): the spatial size of output data after padding.
            method (str): pad image symmetric on every side or only pad at the end sides. default is 'symmetric'.
            mode (str or sequence of str): one of the following string values or a user supplied function:
                {'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric',
                'wrap', 'empty', <function>}
                for more details, please check: https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
        """
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padder = SpatialPad(spatial_size, method)

    def __call__(self, data):
        d = dict(data)
        for key, m in zip(self.keys, self.mode):
            d[key] = self.padder(d[key], mode=m)
        return d


class SpatialCropd(MapTransform):
    """
    dictionary-based wrapper of :py:class:`monai.transforms.SpatialCrop`.
    Either a spatial center and size must be provided, or alternatively if center and size
    are not provided, the start and end coordinates of the ROI must be provided.
    """

    def __init__(self, keys, roi_center=None, roi_size=None, roi_start=None, roi_end=None):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
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
        keys (hashable items): keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size (list, tuple): the size of the crop region e.g. [224,224,128]
    """

    def __init__(self, keys, roi_size):
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
        keys (hashable items): keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size (list, tuple): if `random_size` is True, the spatial size of the minimum crop region.
            if `random_size` is False, specify the expected ROI size to crop. e.g. [224, 224, 128]
        random_center (bool): crop at random position as center or the image center.
        random_size (bool): crop with random size or specific size ROI.
    """

    def __init__(self, keys, roi_size, random_center=True, random_size=True):
        super().__init__(keys)
        self.roi_size = roi_size
        self.random_center = random_center
        self.random_size = random_size

    def randomize(self, img_size):
        self._size = [self.roi_size] * len(img_size) if not isinstance(self.roi_size, (list, tuple)) else self.roi_size
        if self.random_size:
            self._size = [self.R.randint(low=self._size[i], high=img_size[i] + 1) for i in range(len(img_size))]
        if self.random_center:
            valid_size = get_valid_patch_size(img_size, self._size)
            self._slices = ensure_tuple(slice(None)) + get_random_patch(img_size, valid_size, self.R)

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


class CropForegroundd(MapTransform):
    """
    dictionary-based version :py:class:`monai.transforms.CropForeground`.
    Crop only the foreground object of the expected images.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    The valid part can be determined by any field in the data with `source_key`, for example:
    - Select values > 0 in image field as the foreground and crop on all fields specified by `keys`.
    - Select label = 3 in label field as the foreground to crop on all fields specified by `keys`.
    - Select label > 0 in the third channel of a One-Hot label field as the foreground to crop all `keys` fields.
    Users can define arbitrary function to select expected foreground from the whole source image or specified
    channels. And it can also add margin to every dim of the bounding box of foreground object.
    """

    def __init__(self, keys, source_key, select_fn=lambda x: x > 0, channel_indexes=None, margin=0):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            source_key (str): data source to generate the bounding box of foreground, can be image or label, etc.
            select_fn (Callable): function to select expected foreground, default is to select values > 0.
            channel_indexes (int, tuple or list): if defined, select foregound only on the specified channels
                of image. if None, select foreground on the whole image.
            margin (int): add margin to all dims of the bounding box.
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
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys (list): parameter will be used to get and set the actual data item to transform.
        label_key (str): name of key for label image, this will be used for finding foreground/background.
        size (list, tuple): the size of the crop region e.g. [224,224,128]
        pos (int, float): used to calculate the ratio ``pos / (pos + neg)`` for the probability to pick a
          foreground voxel as a center rather than a background voxel.
        neg (int, float): used to calculate the ratio ``pos / (pos + neg)`` for the probability to pick a
          foreground voxel as a center rather than a background voxel.
        num_samples (int): number of samples (crop regions) to take in each list.
        image_key (str): if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold (int or float): if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
    """

    def __init__(self, keys, label_key, size, pos=1, neg=1, num_samples=1, image_key=None, image_threshold=0):
        super().__init__(keys)
        assert isinstance(label_key, str), "label_key must be a string."
        assert isinstance(size, (list, tuple)), "size must be list or tuple."
        assert all(isinstance(x, int) and x > 0 for x in size), "all elements of size must be positive integers."
        assert float(pos) >= 0 and float(neg) >= 0, "pos and neg must be greater than or equal to 0."
        assert float(pos) + float(neg) > 0, "pos and neg cannot both be 0."
        assert isinstance(num_samples, int), "invalid samples number: {}. num_samples must be an integer.".format(
            num_samples
        )
        assert num_samples >= 0, "num_samples must be greater than or equal to 0."
        self.label_key = label_key
        self.size = size
        self.pos_ratio = float(pos) / (float(pos) + float(neg))
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.centers = None

    def randomize(self, label, image):
        self.centers = generate_pos_neg_label_crop_centers(
            label, self.size, self.num_samples, self.pos_ratio, image, self.image_threshold, self.R
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
                    cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.size)
                    results[i][key] = cropper(img)
            else:
                for i in range(self.num_samples):
                    results[i][key] = data[key]

        return results


SpatialPadD = SpatialPadDict = SpatialPadd
SpatialCropD = SpatialCropDict = SpatialCropd
CenterSpatialCropD = CenterSpatialCropDict = CenterSpatialCropd
RandSpatialCropD = RandSpatialCropDict = RandSpatialCropd
CropForegroundD = CropForegroundDict = CropForegroundd
RandCropByPosNegLabelD = RandCropByPosNegLabelDict = RandCropByPosNegLabeld
