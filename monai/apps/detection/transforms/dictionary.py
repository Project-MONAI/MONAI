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
A collection of dictionary-based wrappers around the "vanilla" transforms for box operations
defined in :py:class:`monai.apps.detection.transforms.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from copy import deepcopy
from typing import Dict, Hashable, Mapping, Type, Union

from monai.apps.detection.transforms.array import BoxConvertMode, BoxConvertToStandard
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.box_utils import BoxMode
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform

__all__ = [
    "BoxConvertModed",
    "BoxConvertModeD",
    "BoxConvertModeDict",
    "BoxConvertToStandardd",
    "BoxConvertToStandardD",
    "BoxConvertToStandardDict",
]


class BoxConvertModed(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.BoxConvertMode`.

    This transform converts the boxes in src_mode to the dst_mode.

    Example:
        .. code-block:: python

            data = {"boxes": torch.ones(10,4)}
            # convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
            box_converter = BoxConvertModed(box_keys=["boxes"], src_mode="xyxy", dst_mode="ccwh")
            box_converter(data)
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        src_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
        dst_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            box_keys: Keys to pick data for transformation.
            src_mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
                It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.BoxConvertMode` .
            dst_mode: target box mode. If it is not given, this func will assume it is ``StandardMode()``.
                It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.BoxConvertMode` .
            allow_missing_keys: don't raise exception if key is missing.

        See also :py:class:`monai.apps.detection,transforms.array.BoxConvertMode`
        """
        super().__init__(box_keys, allow_missing_keys)
        self.converter = BoxConvertMode(src_mode=src_mode, dst_mode=dst_mode)
        self.inverse_converter = BoxConvertMode(src_mode=dst_mode, dst_mode=src_mode)

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


class BoxConvertToStandardd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.BoxConvertToStandard`.

    Convert given boxes to standard mode.
    Standard mode is "xyxy" or "xyzxyz",
    representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

    Example:
        .. code-block:: python

            data = {"boxes": torch.ones(10,6)}
            # convert boxes with format [xmin, xmax, ymin, ymax, zmin, zmax] to [xmin, ymin, zmin, xmax, ymax, zmax]
            box_converter = BoxConvertModed(box_keys=["boxes"], mode="xxyyzz")
            box_converter(data)
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        mode: Union[str, BoxMode, Type[BoxMode], None] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            box_keys: Keys to pick data for transformation.
            mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
                It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.BoxConvertMode` .
            allow_missing_keys: don't raise exception if key is missing.

        See also :py:class:`monai.apps.detection,transforms.array.BoxConvertToStandard`
        """
        super().__init__(box_keys, allow_missing_keys)
        self.converter = BoxConvertToStandard(mode=mode)
        self.inverse_converter = BoxConvertMode(src_mode=None, dst_mode=mode)

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


BoxConvertModeD = BoxConvertModeDict = BoxConvertModed
BoxConvertToStandardD = BoxConvertToStandardDict = BoxConvertToStandardd
