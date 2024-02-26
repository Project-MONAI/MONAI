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
A collection of dictionary-based wrappers around the signal operations defined in :py:class:`monai.transforms.signal.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms.signal.array import SignalFillEmpty
from monai.transforms.transform import MapTransform

__all__ = ["SignalFillEmptyd", "SignalFillEmptyD", "SignalFillEmptyDict"]


class SignalFillEmptyd(MapTransform):
    """
    Applies the SignalFillEmptyd transform on the input. All NaN values will be replaced with the
    replacement value.

    Args:
        keys: keys of the corresponding items to model output.
        allow_missing_keys: don't raise exception if key is missing.
        replacement: The value that the NaN entries shall be mapped to.
    """

    backend = SignalFillEmpty.backend

    def __init__(self, keys: KeysCollection = None, allow_missing_keys: bool = False, replacement=0.0):
        super().__init__(keys, allow_missing_keys)
        self.signal_fill_empty = SignalFillEmpty(replacement=replacement)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        for key in self.key_iterator(data):
            data[key] = self.signal_fill_empty(data[key])  # type: ignore

        return data


SignalFillEmptyD = SignalFillEmptyDict = SignalFillEmptyd
