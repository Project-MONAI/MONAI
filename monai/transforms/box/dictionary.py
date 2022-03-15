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
A collection of dictionary-based wrappers around the "vanilla" transforms for spatial operations
defined in :py:class:`monai.transforms.spatial.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from copy import deepcopy
from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

from monai.data.utils import orientation_ras_lps
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import box_utils
from monai.transforms.box.array import BoxClipToImage, BoxConvertMode, BoxConvertToStandard, BoxFlip, BoxAffine, BoxMaskToBox, BoxToBoxMask
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform
from monai.utils.module import optional_import
from monai.utils.enums import PostFix
from monai.utils import (
    ensure_tuple_rep,
)

nib, _ = optional_import("nibabel")

__all__ = [
    "BoxConvertToStandardd",
    "BoxConvertToStandardD",
    "BoxConvertToStandardDict",
    "BoxConvertModed",
    "BoxConvertModeD",
    "BoxConvertModeDict",
    "BoxClipToImaged",
    "BoxClipToImageD",
    "BoxClipToImageDict",
    "BoxFlipd",
    "BoxFlipD",
    "BoxFlipDict",
    "BoxToImageCoordinated",
    "BoxToImageCoordinateD",
    "BoxToImageCoordinateDict",
    "BoxMaskToBoxd",
    "BoxMaskToBoxD",
    "BoxMaskToBoxDict",
    "BoxToBoxMaskd",
    "BoxToBoxMaskD",
    "BoxToBoxMaskDict",
]

DEFAULT_POST_FIX = PostFix.meta()

class BoxConvertToStandardd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.BoxFlip`.

    Args:
        box_keys: Keys to pick data for transformation.
        box_mode: String, if not given, we assume box mode is standard mode
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_mode: str,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        self.converter = BoxConvertToStandard(mode=box_mode)
        self.inverse_converter = BoxConvertMode(mode1=None, mode2=box_mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        key_id = 0
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key])
            key_id += 1
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        key_id = 0
        for key in self.key_iterator(d):
            _ = self.get_most_recent_transform(d, key)
            # Inverse is same as forward
            d[key] = self.inverse_converter(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
            key_id += 1
        return d


class BoxConvertModed(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.BoxFlip`.

    Args:
        box_keys: Keys to pick data for transformation.
        box_mode: String, if not given, we assume box mode is standard mode
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_mode: str,
        tgt_mode: str,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        self.converter = BoxConvertMode(mode1=box_mode, mode2=tgt_mode)
        self.inverse_converter = BoxConvertMode(mode1=tgt_mode, mode2=box_mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        key_id = 0
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key])
            key_id += 1
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        key_id = 0
        for key in self.key_iterator(d):
            _ = self.get_most_recent_transform(d, key)
            # Inverse is same as forward
            d[key] = self.inverse_converter(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
            key_id += 1
        return d


class BoxToImageCoordinated(MapTransform, InvertibleTransform):
    """
    Dictionary-based transfrom that converts box in world coordinate to image coordinate.

    Args:
        box_keys: Keys to pick data for transformation.
        image_key: Key for the images that boxes belong to
        box_mode: String, if not given, we assume box mode is standard mode
        remove_empty: whether to remove the boxes that are actually empty
        allow_missing_keys: don't raise exception if key is missing.
        image_meta_key: explicitly indicate the key of the corresponding meta data dictionary.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the meta data is a dictionary object which contains: filename, affine, original_shape, etc.
            it is a string, map to the `image_key`.
            if None, will try to construct meta_keys by `image_key_{meta_key_postfix}`.
        image_meta_key_postfix: if image_meta_keys=None, use `image_key_{postfix}` to fetch the meta data according
            to the key data, default is `meta_dict`, the meta data is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        lps_reader:bool, whether the image reader reads in image with LPS format, rather than RAS format. True for itkreader, False for nibabel reader.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        image_key: str,
        box_mode: str = None,
        allow_missing_keys: bool = False,
        image_meta_key: str = None,
        image_meta_key_postfix: str = DEFAULT_POST_FIX,
        lps_reader:bool = False
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        self.image_meta_key = image_meta_key or f"{image_key}_{image_meta_key_postfix}"

        if (box_mode is not None) and (box_mode not in ["xxyy","xxyyzz","xyxy","xyzxyz","ccwh", "cccwhd"]):
            raise ValueError(
                f"We support only box mode in [xxyy,xxyyzz,xyxy,xyzxyz,ccwh, cccwhd], bt we got {box_mode}."
            )
        self.lps_reader = lps_reader
        self.converter_to_image_coordinate = BoxAffine(mode=box_mode, invert_affine=True)
        self.converter_to_physical_coordinate = BoxAffine(mode=None, invert_affine=False)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        meta_key = self.image_meta_key
        for key in self.key_iterator(d): 
            # create metadata if necessary
            if meta_key not in d:
                d[meta_key] = {"affine": None}
            meta_data = d[meta_key]

            self.push_transform(d, key)
            
            affine = meta_data["affine"] # RAS affine
            if self.lps_reader: # change it to LPS affine if the reader is LPS
                affine = orientation_ras_lps(affine)
            d[key] = self.converter_to_image_coordinate(d[key], affine=affine)
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        meta_key = self.image_meta_key
        for key in self.key_iterator(d): 
            # create metadata if necessary
            if meta_key not in d:
                d[meta_key] = {"affine": None}
            meta_data = d[meta_key]

            affine = meta_data["affine"] # RAS affine
            if self.lps_reader: # change it to LPS affine if the reader is LPS
                affine = orientation_ras_lps(affine)

            d[key] = self.converter_to_physical_coordinate(d[key], affine=affine)
            self.pop_transform(d, key)
        return d

class BoxClipToImaged(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.BoxFlip`.

    Args:
        box_keys: Keys to pick data for transformation.
        image_key: Key for the images that boxes belong to
        box_mode: String, if not given, we assume box mode is standard mode
        remove_empty: whether to remove the boxes that are actually empty
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        label_keys: KeysCollection,
        image_key: str,
        box_mode: str = None,
        remove_empty: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)

        if (box_mode is not None) and (box_mode not in box_utils.STANDARD_MODE):
            raise ValueError(
                "Currently we support only standard box_mode."
                "Please apply BoxConvertToStandardd first and then set box_mode=None."
            )
        self.label_keys = label_keys
        self.image_key = image_key
        self.clipper = BoxClipToImage(mode=box_mode, remove_empty=remove_empty)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image_size = d[self.image_key].shape[1:]
        key_id = 0
        for key, label_key in self.key_iterator(d, self.label_keys):
            self.push_transform(d, key)
            d[key], d[label_key] = self.clipper(d[key], d[label_key], image_size)
            key_id += 1
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key, label_key in self.key_iterator(d, self.label_keys):
            self.pop_transform(d, key, label_key)
        return d


class BoxFlipd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.
    We suggest performing BoxClipToImaged before this transform.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        image_key: str,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
        box_mode: str = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        if (box_mode is not None) and (box_mode not in box_utils.STANDARD_MODE):
            raise ValueError(
                "Currently we support only standard box_mode."
                "Please apply BoxConvertToStandardd first and then set box_mode=None."
            )
        self.flipper = BoxFlip(spatial_axis=spatial_axis, mode=box_mode)
        self.image_key = image_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image_size = d[self.image_key].shape[1:]
        key_id = 0
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.flipper(d[key], image_size)
            key_id += 1
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        image_size = d[self.image_key].shape[1:]
        key_id = 0
        for key in self.key_iterator(d):
            _ = self.get_most_recent_transform(d, key)
            # Inverse is same as forward
            d[key] = self.flipper(d[key], image_size)
            # Remove the applied transform
            self.pop_transform(d, key)
            key_id += 1
        return d


class BoxToBoxMaskd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.
    Convert box to CxMxNxP mask image, which has the same size with the input image MxNxP.
    The channel number equals to the number of boxes

    Args:
        keys: Keys to pick data for transformation.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        min_label: min foreground box label.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        label_keys: KeysCollection,
        box_mask_keys: KeysCollection,
        image_key: str,
        min_label: int = 0,
        box_mode: str = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        if (box_mode is not None) and (box_mode not in box_utils.STANDARD_MODE):
            raise ValueError(
                "Currently we support only standard box_mode."
                "Please apply BoxConvertToStandardd first and then set box_mode=None."
            )
        self.image_key = image_key
        if len(label_keys) != len(self.keys) or len(box_mask_keys) != len(self.keys):
            raise ValueError("Please make sure len(label_keys)==len(box_keys)==len(box_mask_keys)!")
        self.label_keys = label_keys
        self.box_mask_keys = box_mask_keys
        self.bg_label = min_label - 1
        self.converter = BoxToBoxMask(mode=box_mode, bg_label=self.bg_label)


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image_size = list(d[self.image_key].shape[1:])

        key_id = 0
        for key, label_key, box_mask_key in self.key_iterator(d, self.label_keys,self.box_mask_keys):
            # self.push_transform(d, key)
            # make the mask non-negative
            d[box_mask_key] = self.converter(d[key], image_size, d[label_key]) - self.bg_label
            key_id += 1
        return d

class BoxMaskToBoxd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.
    Convert mask to box

    Args:
        keys: Keys to pick data for transformation.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        min_label: min foreground box label.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        label_keys: KeysCollection,
        box_mask_keys: KeysCollection,
        image_key: str,
        min_label: int = 0,
        box_mode: str = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        if (box_mode is not None) and (box_mode not in box_utils.STANDARD_MODE):
            raise ValueError(
                "Currently we support only standard box_mode."
                "Please apply BoxConvertToStandardd first and then set box_mode=None."
            )
        self.image_key = image_key
        if len(label_keys) != len(self.keys) or len(box_mask_keys) != len(self.keys):
            raise ValueError("Please make sure len(label_keys)==len(box_keys)==len(box_mask_keys)!")
        self.label_keys = label_keys
        self.box_mask_keys = box_mask_keys
        self.bg_label = min_label - 1
        self.converter = BoxMaskToBox(mode=box_mode, bg_label=self.bg_label)


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image_size = d[self.image_key].shape[1:]

        key_id = 0
        for key, label_key, box_mask_key in self.key_iterator(d,self.label_keys,self.box_mask_keys):
            # self.push_transform(d, key)
            # first convert the mask back to the one with bg_label
            d[key], d[label_key] = self.converter(d[box_mask_key] + self.bg_label)
            key_id += 1
        return d





BoxConvertToStandardD = BoxConvertToStandardDict = BoxConvertToStandardd
BoxConvertModeD = BoxConvertModeDict = BoxConvertModed
BoxClipToImageD = BoxClipToImageDict = BoxClipToImaged
BoxFlipD = BoxFlipDict = BoxFlipd
BoxToImageCoordinateD = BoxToImageCoordinateDict= BoxToImageCoordinated
BoxMaskToBoxD = BoxMaskToBoxDict = BoxMaskToBoxd
BoxToBoxMaskD = BoxToBoxMaskDict = BoxToBoxMaskd
