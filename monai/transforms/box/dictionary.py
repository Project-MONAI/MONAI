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

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import box_utils
from monai.transforms.box.array import BoxClipToImage, BoxConvertMode, BoxConvertToStandard, BoxFlip
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform
from monai.utils.module import optional_import

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
]


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
        image_key: str,
        box_mode: str = None,
        remove_empty: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        if (box_mode is not None) and (box_mode not in box_utils.STANDARD_MODE):
            ValueError(
                "Currently we support only standard box_mode."
                "Please apply BoxConvertToStandardd first and then set box_mode=None."
            )
        self.image_key = image_key
        self.clipper = BoxClipToImage(mode=box_mode, remove_empty=remove_empty)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image_size = d[self.image_key].shape[1:]
        key_id = 0
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.clipper(d[key], image_size)
            key_id += 1
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        return d


class BoxFlipd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.

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
        remove_empty: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        if (box_mode is not None) and (box_mode not in box_utils.STANDARD_MODE):
            ValueError(
                "Currently we support only standard box_mode."
                "Please apply BoxConvertToStandardd first and then set box_mode=None."
            )
        self.flipper = BoxFlip(spatial_axis=spatial_axis, mode=box_mode, remove_empty=remove_empty)
        self.image_key = image_key
        self.remove_empty = remove_empty

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


BoxConvertToStandardD = BoxConvertToStandardDict = BoxConvertToStandardd
BoxConvertModeD = BoxConvertModeDict = BoxConvertModed
BoxClipToImageD = BoxClipToImageDict = BoxClipToImaged
BoxFlipD = BoxFlipDict = BoxFlipd
