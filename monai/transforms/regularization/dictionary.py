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

from __future__ import annotations

from monai.config import KeysCollection
from monai.utils.misc import ensure_tuple

from ..transform import MapTransform
from .array import CutMix, CutOut, MixUp

__all__ = ["MixUpd", "MixUpD", "MixUpDict", "CutMixd", "CutMixD", "CutMixDict", "CutOutd", "CutOutD", "CutOutDict"]


class MixUpd(MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.MixUp`.

    Notice that the mixup transformation will be the same for all entries
    for consistency, i.e. images and labels must be applied the same augmenation.
    """

    def __init__(
        self, keys: KeysCollection, batch_size: int, alpha: float = 1.0, allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mixup = MixUp(batch_size, alpha)

    def __call__(self, data):
        self.mixup.randomize()
        result = dict(data)
        for k in self.keys:
            result[k] = self.mixup.apply(data[k])
        return result


class CutMixd(MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.CutMix`.

    Notice that the mixture weights will be the same for all entries
    for consistency, i.e. images and labels must be aggregated with the same weights,
    but the random crops are not.
    """

    def __init__(
        self,
        keys: KeysCollection,
        batch_size: int,
        label_keys: KeysCollection | None = None,
        alpha: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mixer = CutMix(batch_size, alpha)
        self.label_keys = ensure_tuple(label_keys) if label_keys is not None else []

    def __call__(self, data):
        self.mixer.randomize()
        result = dict(data)
        for k in self.keys:
            result[k] = self.mixer.apply(data[k])
        for k in self.label_keys:
            result[k] = self.mixer.apply_on_labels(data[k])
        return result


class CutOutd(MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.CutOut`.

    Notice that the cutout is different for every entry in the dictionary.
    """

    def __init__(self, keys: KeysCollection, batch_size: int, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.cutout = CutOut(batch_size)

    def __call__(self, data):
        result = dict(data)
        self.cutout.randomize()
        for k in self.keys:
            result[k] = self.cutout(data[k])
        return result


MixUpD = MixUpDict = MixUpd
CutMixD = CutMixDict = CutMixd
CutOutD = CutOutDict = CutOutd
