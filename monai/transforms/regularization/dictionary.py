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

from collections.abc import Hashable

import numpy as np

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.utils import convert_to_tensor
from monai.utils.misc import ensure_tuple

from ..transform import MapTransform, RandomizableTransform
from .array import CutMix, CutOut, MixUp

__all__ = ["MixUpd", "MixUpD", "MixUpDict", "CutMixd", "CutMixD", "CutMixDict", "CutOutd", "CutOutD", "CutOutDict"]


class MixUpd(MapTransform, RandomizableTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.MixUp`.

    Notice that the mixup transformation will be the same for all entries
    for consistency, i.e. images and labels must be applied the same augmenation.
    """

    def __init__(
        self, keys: KeysCollection, batch_size: int, alpha: float = 1.0, allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.mixup = MixUp(batch_size, alpha)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> MixUpd:
        super().set_random_state(seed, state)
        self.mixup.set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        # all the keys share the same random state
        self.mixup.randomize(None)
        for k in self.key_iterator(d):
            d[k] = self.mixup(data[k], randomize=False)
        return d


class CutMixd(MapTransform, RandomizableTransform):
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

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> CutMixd:
        super().set_random_state(seed, state)
        self.mixer.set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, NdarrayOrTensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out
        self.mixer.randomize(d[first_key])
        for key, label_key in self.key_iterator(d, self.label_keys):
            ret = self.mixer(data[key], data.get(label_key, None), randomize=False)
            d[key] = ret[0]
            if label_key in d:
                d[label_key] = ret[1]
        return d


class CutOutd(MapTransform, RandomizableTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.CutOut`.

    Notice that the cutout is different for every entry in the dictionary.
    """

    def __init__(self, keys: KeysCollection, batch_size: int, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.cutout = CutOut(batch_size)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> CutOutd:
        super().set_random_state(seed, state)
        self.cutout.set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out: dict[Hashable, NdarrayOrTensor] = convert_to_tensor(d, track_meta=get_track_meta())
            return out
        self.cutout.randomize(d[first_key])
        for k in self.key_iterator(d):
            d[k] = self.cutout(data[k], randomize=False)
        return d


MixUpD = MixUpDict = MixUpd
CutMixD = CutMixDict = CutMixd
CutOutD = CutOutDict = CutOutd
